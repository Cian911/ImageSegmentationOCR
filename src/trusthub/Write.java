package trusthub;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 * Created by Cian on 11/8/2015.
 */
class Write {

    private Mat writeSrcImage = new Mat();
    private String imgPathToWrite = "";

    public Write (Mat writeSrcImage, String imgPathToWrite) {
        this.writeSrcImage = writeSrcImage;
        this.imgPathToWrite = imgPathToWrite;
    }

    public Mat returnWrittenImage () {
        return writeSrcImage;
    }

    public void writeImageToLocation () {
        Imgcodecs.imwrite(this.imgPathToWrite, this.writeSrcImage);
    }

}
