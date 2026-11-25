package jp.keireality.camera;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

final class PhoneModel implements AutoCloseable {
    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession session;
    private final ByteBuffer rig = ByteBuffer.allocateDirect(2*3*160*320).order(ByteOrder.nativeOrder());
    private final ByteBuffer mask = ByteBuffer.allocateDirect(2*80*160).order(ByteOrder.nativeOrder());
    private final int[] colors = new int[384*192];
    private RigSampler sampler;

    PhoneModel(AssetManager assets) throws IOException, OrtException {
        try (AssetFileDescriptor fd=assets.openFd("keireality.onnx");
             FileInputStream input=fd.createInputStream();
             OrtSession.SessionOptions options=new OrtSession.SessionOptions()) {
            ByteBuffer weights=input.getChannel().map(FileChannel.MapMode.READ_ONLY,fd.getStartOffset(),fd.getLength());
            options.setIntraOpNumThreads(3);
            options.setInterOpNumThreads(1);
            session=env.createSession(weights,options);
        }
    }

    Bitmap render(CameraFrames.Frame frame, double focalPixels) throws OrtException {
        long f=Math.round(focalPixels);
        if (sampler==null || frame.width!=sampler.width || frame.height!=sampler.height || sampler.focal!=f) {
            sampler=new RigSampler(frame.width,frame.height,f);
        }
        rig.clear(); rig.put(sampler.sample(frame.argb)).rewind();
        mask.clear(); mask.put(sampler.visible).rewind();
        try (OnnxTensor image=OnnxTensor.createTensor(env,rig,new long[]{1,2,3,160,320},OnnxJavaType.UINT8);
             OnnxTensor visible=OnnxTensor.createTensor(env,mask,new long[]{1,2,80,160},OnnxJavaType.UINT8)) {
            Map<String,OnnxTensor> inputs=new HashMap<>();
            inputs.put("rig",image);
            inputs.put("visible",visible);
            try (OrtSession.Result result=session.run(inputs)) {
                ByteBuffer out=((OnnxTensor)result.get(0)).getByteBuffer();
                if (out.remaining()!=3*colors.length) throw new IllegalStateException("生成画像の寸法が異なります");
                for (int i=0;i<colors.length;i++) {
                    colors[i]=0xff000000 | (out.get()&255)<<16 | (out.get()&255)<<8 | (out.get()&255);
                }
            }
        }
        return Bitmap.createBitmap(colors,384,192,Bitmap.Config.ARGB_8888);
    }

    @Override public void close() throws OrtException { session.close(); }
}
