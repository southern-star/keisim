package jp.keireality.camera;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.hardware.camera2.*;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.view.WindowInsets;
import android.view.WindowInsetsController;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Locale;

public final class CameraActivity extends Activity {
    private final HandlerThread worker = new HandlerThread("KeiReality camera");
    private Handler handler;
    private Button start;
    private TextView status;
    private ImageView generated;
    private View controls;
    private boolean active, resumed, startAfterGrant;
    private volatile long generation;
    // The camera, reader and model are owned by the same worker.
    private CameraDevice device;
    private CameraCaptureSession capture;
    private ImageReader reader;
    private PhoneModel model;
    private float focalMm;
    private Rect captureCrop;
    private long firstOutput;
    private int count;

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        setContentView(R.layout.camera);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            getWindow().setDecorFitsSystemWindows(false);
        }
        controls=findViewById(R.id.controls);
        controls.setOnApplyWindowInsetsListener((v,insets)->{
            v.setPadding(insets.getSystemWindowInsetLeft(), 0,
                         insets.getSystemWindowInsetRight(), insets.getSystemWindowInsetBottom());
            return insets;
        });
        start=findViewById(R.id.start); status=findViewById(R.id.status);
        generated=findViewById(R.id.generated);
        generated.setOnClickListener(v -> {
            if (generated.getDrawable() != null) {
                showControls(controls.getVisibility() != View.VISIBLE);
            }
        });
        showControls(true);
        worker.start(); handler=new Handler(worker.getLooper());
        start.setOnClickListener(v->{
            if (active) { stop("停止しました。"); return; }
            if (checkSelfPermission(Manifest.permission.CAMERA)!=PackageManager.PERMISSION_GRANTED) {
                requestPermissions(new String[]{Manifest.permission.CAMERA},1);
            } else begin();
        });
    }

    private void showControls(boolean visible) {
        controls.setVisibility(visible ? View.VISIBLE : View.GONE);
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            WindowInsetsController bars = getWindow().getInsetsController();
            if (bars == null) return;
            bars.setSystemBarsBehavior(WindowInsetsController.BEHAVIOR_SHOW_TRANSIENT_BARS_BY_SWIPE);
            if (visible) bars.show(WindowInsets.Type.systemBars());
            else bars.hide(WindowInsets.Type.systemBars());
        } else {
            int flags = View.SYSTEM_UI_FLAG_LAYOUT_STABLE | View.SYSTEM_UI_FLAG_LAYOUT_FULLSCREEN
                | View.SYSTEM_UI_FLAG_LAYOUT_HIDE_NAVIGATION;
            if (!visible) flags |= View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION
                | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
            getWindow().getDecorView().setSystemUiVisibility(flags);
        }
    }

    @Override public void onRequestPermissionsResult(int request,String[] permissions,int[] results) {
        super.onRequestPermissionsResult(request,permissions,results);
        if (request!=1) return;
        if (results.length>0 && results[0]==PackageManager.PERMISSION_GRANTED) {
            if (resumed) begin(); else startAfterGrant=true;
        } else status.setText("カメラへのアクセスを許可すると開始できます。");
    }

    @Override public void onResume() {
        super.onResume(); resumed=true;
        if (startAfterGrant) { startAfterGrant=false; begin(); }
    }

    @Override public void onPause() {
        resumed=false;
        if (active) stop("停止しました。");
        super.onPause();
    }

    private void begin() {
        if (active || !resumed) return;
        active=true;
        long ticket=++generation;
        start.setText("停止"); status.setText("端末内のモデルとカメラを準備しています…");
        handler.post(()->{
            try {
                if (model==null) model=new PhoneModel(getAssets());
                if (ticket==generation) openCamera(ticket);
            } catch (Exception error) { fail(ticket,error); }
        });
    }

    private void stop(String message) {
        ++generation; active=false;
        start.setText("カメラを開始"); status.setText(message);
        generated.setImageDrawable(null);
        showControls(true);
        handler.post(this::closeCamera);
    }

    private void fail(long ticket,Exception error) {
        Log.e("KeiReality","Camera display failed",error);
        runOnUiThread(()->{ if (ticket==generation) stop("開始・生成に失敗しました。" + error.getLocalizedMessage()); });
    }

    private void closeCamera() {
        if (capture!=null) { capture.close(); capture=null; }
        if (device!=null) { device.close(); device=null; }
        if (reader!=null) { reader.close(); reader=null; }
    }

    @Override public void onDestroy() {
        ++generation;
        handler.post(()->{
            closeCamera();
            try { if (model!=null) model.close(); }
            catch (Exception e) { Log.w("KeiReality","Closing model",e); }
            worker.quitSafely();
        });
        super.onDestroy();
    }

    private void openCamera(long ticket) throws Exception {
        CameraManager manager=getSystemService(CameraManager.class);
        for (String id:manager.getCameraIdList()) {
            CameraCharacteristics camera=manager.getCameraCharacteristics(id);
            if (!Integer.valueOf(CameraCharacteristics.LENS_FACING_BACK).equals(camera.get(CameraCharacteristics.LENS_FACING))) continue;
            StreamConfigurationMap config=camera.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
            Size[] sizes=config==null?null:config.getOutputSizes(ImageFormat.YUV_420_888);
            if (sizes==null || sizes.length==0) continue;
            Size size=Arrays.stream(sizes).min(Comparator.comparingLong(s->
                Math.abs((long)s.getWidth()*s.getHeight()-640L*480))).orElseThrow();
            ImageReader images=ImageReader.newInstance(size.getWidth(),size.getHeight(),ImageFormat.YUV_420_888,3);
            reader=images;
            float[] lenses=camera.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS);
            focalMm=lenses!=null && lenses.length>0 ? lenses[0] : 0;
            captureCrop=null; firstOutput=0; count=0;
            images.setOnImageAvailableListener(r->process(ticket,r,camera),handler);
            manager.openCamera(id,new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice opened) {
                    if (ticket!=generation) { opened.close(); return; }
                    device=opened;
                    try { configure(ticket,opened,images); } catch (Exception e) { fail(ticket,e); }
                }
                @Override public void onDisconnected(CameraDevice opened) {
                    opened.close(); fail(ticket,new IllegalStateException("カメラが切断されました"));
                }
                @Override public void onError(CameraDevice opened,int code) {
                    opened.close(); fail(ticket,new IllegalStateException("カメラエラー "+code));
                }
            },handler);
            return;
        }
        throw new IllegalStateException("背面カメラを利用できません");
    }

    private void configure(long ticket,CameraDevice opened,ImageReader images) throws CameraAccessException {
        CaptureRequest.Builder request=opened.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
        request.addTarget(images.getSurface());
        opened.createCaptureSession(Collections.singletonList(images.getSurface()),new CameraCaptureSession.StateCallback() {
            @Override public void onConfigured(CameraCaptureSession session) {
                if (ticket!=generation) { session.close(); return; }
                capture=session;
                try {
                    session.setRepeatingRequest(request.build(),new CameraCaptureSession.CaptureCallback() {
                        @Override public void onCaptureCompleted(CameraCaptureSession s,CaptureRequest req,TotalCaptureResult result) {
                            if (ticket!=generation) return;
                            Float length=result.get(CaptureResult.LENS_FOCAL_LENGTH);
                            if (length!=null) focalMm=length;
                            captureCrop=result.get(CaptureResult.SCALER_CROP_REGION);
                        }
                    },handler);
                } catch (Exception e) { fail(ticket,e); }
            }
            @Override public void onConfigureFailed(CameraCaptureSession s) {
                s.close(); fail(ticket,new IllegalStateException("カメラ映像を構成できません"));
            }
        },handler);
    }

    private void process(long ticket,ImageReader images,CameraCharacteristics camera) {
        if (ticket!=generation) return;
        try {
            CameraFrames.Frame frame;
            double focal;
            try (Image image=images.acquireLatestImage()) {
                if (image==null) return;
                int angle=getWindowManager().getDefaultDisplay().getRotation()*90;
                Integer orientation=camera.get(CameraCharacteristics.SENSOR_ORIENTATION);
                int rotation=((orientation==null?0:orientation)-angle+360)%360;
                frame=CameraFrames.read(image,rotation);
                focal=CameraFrames.focalPixels(camera,focalMm,captureCrop,image.getWidth(),image.getHeight());
            }
            long before=SystemClock.elapsedRealtimeNanos();
            Bitmap rendered=model.render(frame,focal);
            long now=SystemClock.elapsedRealtimeNanos();
            if (count++==0) firstOutput=now;
            String measurement=String.format(Locale.JAPAN,"%d コマ　%.0f ms / 推論",count,(now-before)/1e6);
            if (count>1) measurement+=String.format(Locale.JAPAN,"　%.1f fps",(count-1)*1e9/(now-firstOutput));
            final String line=measurement;
            runOnUiThread(()->{
                if (ticket!=generation) return;
                boolean firstFrame = generated.getDrawable() == null;
                generated.setImageBitmap(rendered);
                status.setText(line);
                if (firstFrame) showControls(false);
            });
        } catch (Exception error) { fail(ticket,error); }
    }
}
