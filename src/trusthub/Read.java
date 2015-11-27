package trusthub;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

/**
 * Created by Cian on 11/8/2015.
 */
class Read {

    private Mat readSrcImage;

    public void readImageFromSrc () {
        this.readSrcImage = Imgcodecs.imread("C://Users//Cian//IdeaProjects//TrustHub//src//assets//IMG_3902.jpg");
    }

    public Mat returnSrcImage () {
        return this.readSrcImage;
    }
}
