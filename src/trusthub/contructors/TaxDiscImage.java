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
  private List<MatOfPoint> countors = new ArrayList<MatOfPoint>();

  private int image_size;

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

    KeyPoint kpoint = new KeyPoint();

    Mat mask = applyMatrixMask();

    Scalar zeros = new Scalar(0, 0, 0);

    Mat kernel = createKernalMatrix(1, 50, 255);
    Mat hierarchy = new Mat();

    this.image_size = src_image.height() * src_image.width();

    this.keypoint = applyMSERFeatureDector( binary_image, this.keypoint );
    this.listpoint = drawMSERKeypointsFound( binary_image, this.keypoint );

    mask = determineRegionOfInterest( binary_image, mask, kpoint );

  }

  public Mat determineRegionOfInterest( Mat binary_image, Mat mask, KeyPoint keypoint ) {

    int rectanx1;
    int rectany1;
    int rectanx2;
    int rectany2;
    int scalar_value = 255;

    for( int i = 0; i < this.listpoint.size(); i++ ) {
      kpoint = this.listpoint.get(ind);
      rectanx1 = (int) (kpoint.pt.x - 0.5 * kpoint.size);
      rectany1 = (int) (kpoint.pt.y - 0.5 * kpoint.size);
      rectanx2 = (int) (kpoint.size);
      rectany2 = (int) (kpoint.size);
      if (rectanx1 <= 0)
          rectanx1 = 1;
      if (rectany1 <= 0)
          rectany1 = 1;
      if ((rectanx1 + rectanx2) > binary_image.width())
          rectanx2 = binary_image.width() - rectanx1;
      if ((rectany1 + rectany2) > binary_image.height())
          rectany2 = binary_image.height() - rectany1;

      Rect rectangle_region = new Rect(rectanx1, rectany1, rectanx2, rectany2);
      Mat roi = new Mat(mask, rectangle_region);

      roi.setTo(new Scalar( scalar_value ));
    }

    return mask;
  }

  public ArrayList<Mat> getExtractedImages() {
    return this.getAllExtractedMatrices();
  }

}
