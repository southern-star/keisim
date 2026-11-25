package jp.keireality.camera;

import android.accessibilityservice.AccessibilityService;
import android.app.Activity;
import android.app.Instrumentation;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.os.SystemClock;
import android.provider.Settings;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.json.JSONObject;
import java.io.File;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Supplier;

/** Run on an offline emulator with a rear camera, camera permission and the fullscreen tutorial acknowledged. */
public final class CameraCheck extends Instrumentation {
    private Activity activity;

    @Override public void onCreate(Bundle arguments) { super.onCreate(arguments); start(); }

    private <T> T ui(Supplier<T> read) {
        AtomicReference<T> value=new AtomicReference<>();
        runOnMainSync(()->value.set(read.get()));
        return value.get();
    }

    private Button button() { return activity.findViewById(R.id.start); }
    private String status() {
        return ui(()->((TextView)activity.findViewById(R.id.status)).getText().toString());
    }

    private int frames(String label) {
        int split=label.indexOf(" コマ");
        return split<0 ? 0 : Integer.parseInt(label.substring(0,split));
    }

    private String awaitFrames(int minimum) {
        long end=SystemClock.elapsedRealtime()+75000;
        String label="";
        while (SystemClock.elapsedRealtime()<end) {
            label=status();
            if (frames(label)>=minimum) return label;
            if (label.contains("失敗")) throw new AssertionError(label);
            SystemClock.sleep(100);
        }
        throw new AssertionError("Camera timed out: "+label);
    }

    private ImageView video() { return activity.findViewById(R.id.generated); }

    private void expectControls(boolean visible) {
        if (ui(() -> activity.findViewById(R.id.controls).getVisibility() == View.VISIBLE) != visible) {
            throw new AssertionError("Unexpected playback controls visibility");
        }
    }

    private void tapVideo() {
        int[] center = ui(() -> {
            ImageView image = video();
            int[] point = new int[2];
            image.getLocationOnScreen(point);
            point[0] += image.getWidth() / 2;
            point[1] += image.getHeight() / 2;
            return point;
        });
        long down = SystemClock.uptimeMillis();
        for (int action : new int[]{MotionEvent.ACTION_DOWN, MotionEvent.ACTION_UP}) {
            MotionEvent event = MotionEvent.obtain(down, SystemClock.uptimeMillis(), action, center[0], center[1], 0);
            try {
                sendPointerSync(event);
            } finally {
                event.recycle();
            }
        }
    }

    private String checkFullscreenImage() {
        return ui(() -> {
            ImageView image = video();
            View screen = activity.findViewById(R.id.screen);
            int width = screen.getWidth(), height = screen.getHeight();
            if (image.getWidth() != width || image.getHeight() != height) return "";
            RectF bounds = new RectF(0, 0, image.getDrawable().getIntrinsicWidth(), image.getDrawable().getIntrinsicHeight());
            image.getImageMatrix().mapRect(bounds);
            float expectedWidth = Math.min(width, height * 2f);
            if (Math.abs(bounds.width() - expectedWidth) > 1 || Math.abs(bounds.height() - expectedWidth / 2) > 1
                    || Math.abs(bounds.centerX() - width / 2f) > 1 || Math.abs(bounds.centerY() - height / 2f) > 1)
                return "";
            return width + "x" + height;
        });
    }

    private void assertImage(Bitmap image) {
        long sum=0, squared=0, n=0;
        for (int y=0;y<image.getHeight();y+=4) for (int x=0;x<image.getWidth();x+=4) {
            int pixel=image.getPixel(x,y), green=(pixel>>8)&255;
            sum+=green; squared+=green*green; n++;
        }
        if (squared/(double)n-Math.pow(sum/(double)n,2)<20) throw new AssertionError("Blank image");
    }

    private void checkStopped() {
        expectControls(true);
        if (!ui(()->button().getText().toString()).equals("カメラを開始")
                || ui(()->((ImageView)activity.findViewById(R.id.generated)).getDrawable())!=null)
            throw new AssertionError("Camera did not stop");
    }

    @Override public void onStart() {
        Bundle output=new Bundle();
        try {
            Context context=getTargetContext();
            if (Settings.Global.getInt(context.getContentResolver(),Settings.Global.AIRPLANE_MODE_ON,0)!=1)
                throw new AssertionError("Run this check on the offline test emulator");
            Intent launch=new Intent(context,CameraActivity.class).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
            activity=startActivitySync(launch);
            expectControls(true);
            runOnMainSync(()->button().performClick());
            String first=awaitFrames(4);
            long start=SystemClock.elapsedRealtimeNanos();
            String last=awaitFrames(frames(first)+28);
            double fps=(frames(last)-frames(first))*1e9/(SystemClock.elapsedRealtimeNanos()-start);
            Bitmap generated = ui(() -> ((BitmapDrawable)video().getDrawable()).getBitmap());
            assertImage(generated);
            if (generated.getWidth()!=384 || generated.getHeight()!=192) throw new AssertionError("Wrong output dimensions");
            expectControls(false);
            String viewport = checkFullscreenImage();
            if (viewport.isEmpty()) throw new AssertionError("Image does not fill the viewport at its native aspect ratio");
            File dir=context.getCacheDir();
            try (FileOutputStream file=new FileOutputStream(new File(dir,"camera-check.png"))) {
                getUiAutomation().takeScreenshot().compress(Bitmap.CompressFormat.PNG,100,file);
            }
            tapVideo();
            expectControls(true);
            awaitFrames(frames(status()) + 2);
            tapVideo();
            expectControls(false);
            tapVideo();
            expectControls(true);
            runOnMainSync(()->button().performClick());
            checkStopped();
            runOnMainSync(()->button().performClick());
            String restarted=awaitFrames(3);
            expectControls(false);
            if (!getUiAutomation().performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME)) throw new AssertionError("Home failed");
            long deadline=SystemClock.elapsedRealtime()+5000;
            while (!ui(()->button().getText().toString()).equals("カメラを開始") && SystemClock.elapsedRealtime()<deadline)
                SystemClock.sleep(50);
            checkStopped();
            JSONObject result=new JSONObject().put("airplane_mode",true).put("camera","Android virtualscene")
                .put("abi",android.os.Build.SUPPORTED_ABIS[0]).put("api",android.os.Build.VERSION.SDK_INT)
                .put("first",first).put("last",last).put("measured_display_fps",fps)
                .put("viewport", viewport).put("generated_size","384x192").put("tap_toggles_controls",true)
                .put("restart",restarted).put("stops_in_background",true);
            try (FileOutputStream file=new FileOutputStream(new File(dir,"camera-check.json"))) {
                file.write(result.toString(2).getBytes(StandardCharsets.UTF_8));
            }
            output.putString("result",result.toString());
            finish(Activity.RESULT_OK,output);
        } catch (Throwable error) {
            output.putString("error",android.util.Log.getStackTraceString(error));
            finish(Activity.RESULT_CANCELED,output);
        }
    }
}
