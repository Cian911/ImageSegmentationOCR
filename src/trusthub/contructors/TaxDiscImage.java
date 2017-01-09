package trusthub.contructors;

import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfKeyPoint;
import trusthub.abstractions.ImageManipulation;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Cian on 1/9/2017.
 */
public class TaxDiscImage extends ImageManipulation {

  private Mat src_image;
  private MatOfKeyPoint keypoint = new MatOfKeyPoint();
  private List<KeyPoint> listpoint = new ArrayList<KeyPoint>();

  public TaxDiscImage( Mat src_image, double area_size ) {
    this.setAreaSize(area_size);
    this.src_image = src_image;
  }

  public void performImageSegmentation( Mat src_image ) {
    src_image = cloneImage(src_image);
    Mat binary_image = new Mat();

    if( src_image.channels() > 1 ) {
      binary_image = binarizeColouredImage(src_image);
    } else {
      binary_image = src_image.clone();
    }

    binary_image = applyCannyEdgeDectection( binary_image, 10, 10, 3 );
  }

  public ArrayList<Mat> getExtractedImages() {
    return this.getAllExtractedMatrices();
  }

}
