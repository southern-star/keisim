package jp.keireality.camera;

/** Reproject a level rectilinear rear camera into the training rig, using pixel-centre coordinates. */
final class RigSampler {
    static final int W=320, H=160, PLANE=W*H;
    final int width, height;
    final long focal;
    final byte[] visible = new byte[2*(H/2)*(W/2)];
    private final int[] index = new int[2*PLANE];
    private final float[] wx = new float[index.length], wy = new float[index.length];
    final byte[] rgb = new byte[2*3*PLANE];

    RigSampler(int width, int height, long focal) {
        this.width=width; this.height=height; this.focal=focal;
        double wideFocal = W / Math.PI, narrowFocal = (W/2.0) / Math.tan(Math.toRadians(25));
        double wideTilt=Math.tan(Math.toRadians(8)), cp=Math.cos(Math.toRadians(2)), sp=Math.sin(Math.toRadians(2));
        for (int camera=0; camera<2; camera++) {
            for (int y=0; y<H; y++) for (int x=0; x<W; x++) {
                double right, down, forward;
                if (camera==0) {
                    double az=(x-W/2.0)/wideFocal;
                    right=Math.sin(az); down=(y-H/2.0)/wideFocal+wideTilt; forward=Math.cos(az);
                } else {
                    double b=(y+.5-H/2.0)/narrowFocal;
                    right=(x+.5-W/2.0)/narrowFocal; down=b*cp+sp; forward=cp-b*sp;
                }
                double u=width/2.0-.5+focal*right/forward, v=height/2.0-.5+focal*down/forward;
                int i=camera*PLANE+y*W+x;
                boolean valid=forward>1e-6 && u>-.5 && u<width-.5 && v>-.5 && v<height-.5;
                if (!valid) { index[i]=-1; continue; }
                u=Math.max(0,Math.min(width-1,u)); v=Math.max(0,Math.min(height-1,v));
                int left=Math.min((int)u,width-2), top=Math.min((int)v,height-2);
                index[i]=top*width+left; wx[i]=(float)(u-left); wy[i]=(float)(v-top);
                if ((x&1)==0 && (y&1)==0) visible[(camera*H/2+y/2)*(W/2)+x/2]=1;
            }
        }
    }

    byte[] sample(int[] pixels) {
        for (int i=0;i<index.length;i++) {
            int p=index[i], dest=(i/PLANE)*3*PLANE+i%PLANE;
            if (p<0) {
                rgb[dest]=124; rgb[dest+PLANE]=116; rgb[dest+2*PLANE]=104;
                continue;
            }
            int a=pixels[p], b=pixels[p+1], c=pixels[p+width], d=pixels[p+width+1];
            float k=wx[i], l=wy[i];
            for (int shift=16;shift>=0;shift-=8, dest+=PLANE) {
                int ca=(a>>shift)&255, cb=(b>>shift)&255, cc=(c>>shift)&255, cd=(d>>shift)&255;
                float upper=ca+k*(cb-ca), lower=cc+k*(cd-cc);
                rgb[dest]=(byte)Math.round(upper+l*(lower-upper));
            }
        }
        return rgb;
    }
}
