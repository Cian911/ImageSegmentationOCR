package trusthub;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * Created by Cian on 11/8/2015.
 */
class Read {

    private Mat readSrcImage;

    public void readImageFromSrc (String imageSrc) {
        System.out.println(imageSrc);
        this.readSrcImage = Imgcodecs.imread(imageSrc);
    }

    public Mat returnSrcImage () {
        return this.readSrcImage;
    }
}