package jp.keireality.camera;

import android.graphics.Rect;
import android.hardware.camera2.CameraCharacteristics;
import android.media.Image;
import android.util.Size;
import android.util.SizeF;

import java.nio.ByteBuffer;

final class CameraFrames {
    private CameraFrames() {}

    static double focalPixels(CameraCharacteristics camera, float focalMm, Rect captureCrop, int width, int height) {
        SizeF sensor = camera.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE);
        Size pixels = camera.get(CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE);
        Rect crop = captureCrop != null ? captureCrop : camera.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE);
        if (sensor == null || pixels == null || crop == null || sensor.getWidth() <= 0 || focalMm <= 0
                || crop.width() <= 0 || crop.height() <= 0) {
            return Math.max(width,height) / (2 * Math.tan(Math.toRadians(75) / 2));
        }
        double streamCropWidth = Math.min(crop.width(), crop.height() * width / (double) height);
        return focalMm * pixels.getWidth() * width / (sensor.getWidth() * streamCropWidth);
    }

    private static int clip(int value) {
        return Math.max(0, Math.min(255, value));
    }

    static Frame read(Image image, int rotation) {
        Rect crop = image.getCropRect();
        int w = crop.width(), h = crop.height();
        Frame frame = new Frame(rotation % 180 == 0 ? w : h, rotation % 180 == 0 ? h : w);
        Image.Plane[] planes = image.getPlanes();
        ByteBuffer y = planes[0].getBuffer(), u = planes[1].getBuffer(), v = planes[2].getBuffer();
        int ys=planes[0].getPixelStride(), us=planes[1].getPixelStride(), vs=planes[2].getPixelStride();
        for (int row=0; row<h; row++) {
            int sy = crop.top + row;
            int yRow = y.position() + sy * planes[0].getRowStride();
            int uRow = u.position() + (sy/2) * planes[1].getRowStride();
            int vRow = v.position() + (sy/2) * planes[2].getRowStride();
            int out, step;
            switch (rotation) {
                case 90: out=h-1-row; step=h; break;
                case 180: out=w*(h-row)-1; step=-1; break;
                case 270: out=(w-1)*h+row; step=-h; break;
                default: out=row*w; step=1;
            }
            for (int col=0; col<w; col++, out+=step) {
                int sx = crop.left + col;
                int lum = 298 * Math.max(0, (y.get(yRow+sx*ys)&255) - 16);
                int cb = (u.get(uRow+(sx/2)*us)&255) - 128;
                int cr = (v.get(vRow+(sx/2)*vs)&255) - 128;
                int red=clip((lum+409*cr+128)>>8), green=clip((lum-100*cb-208*cr+128)>>8);
                int blue=clip((lum+516*cb+128)>>8);
                frame.argb[out] = 0xff000000 | red<<16 | green<<8 | blue;
            }
        }
        return frame;
    }

    static final class Frame {
        final int width, height;
        final int[] argb;
        Frame(int width, int height) {
            this.width=width; this.height=height; argb=new int[width*height];
        }
    }
}
