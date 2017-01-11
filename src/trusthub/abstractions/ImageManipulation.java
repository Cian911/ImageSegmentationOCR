package trusthub.abstractions;

import org.opencv.core.*;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;


/**
 * Created by Cian on 1/9/2017.
 */
public abstract class ImageManipulation {

  private double areaSize;
  private ArrayList<Mat> extracted_images = new ArrayList<Mat>();
  private Mat src_image;

  protected Mat binarizeColouredImage( Mat src_image ) {
    Mat binary_matrix = new Mat();
    Imgproc.cvtColor(src_image, binary_matrix, Imgproc.COLOR_RGB2GRAY);

    return binary_matrix;
  }

  protected Mat applyCannyEdgeDectection( Mat src_image, int threshold1, int threshold2, int stroke ) {
    Imgproc.Canny(src_image, src_image, threshold1, threshold2, stroke, true);

    return src_image;
  }

  protected MatOfKeyPoint applyMSERFeatureDector( Mat src_image, MatOfKeyPoint keypoint ) {
    FeatureDetector detector = FeatureDetector
            .create(FeatureDetector.MSER);
    detector.detect(src_image, keypoint);

    return keypoint;
  }

  protected Mat createKernalMatrix( int threshold1, int threshold2, int scalar_value ) {
    Mat kernel = new Mat(threshold1, threshold2, CvType.CV_8UC1, Scalar.all(scalar_value));
    return kernel;
  }

  protected Mat applyMatrixMask( Mat src_image ) {
    Mat mask = new Mat( src_image.size(), CvType.CV_8UC1 );
    return mask;
  }

  protected List<KeyPoint> drawMSERKeypointsFound( Mat src_image, MatOfKeyPoint keypoint ) {
    Features2d.drawKeypoints(src_image, keypoint, src_image);

    return keypoint.toList();
  }

  protected Mat applyMorphologicalOperation( Mat mask, Mat kernel ) {
    Mat morbyte = new Mat();
    Imgproc.morphologyEx(mask, morbyte, Imgproc.MORPH_DILATE, kernel);

    return morbyte;
  }

  protected List<MatOfPoint> findExtractedImageContours( Mat morbyte, List<MatOfPoint> contours, Mat hierarchy ) {
    Imgproc.findContours(morbyte, contours, hierarchy,
            Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

    return contours;
  }

  protected Mat deskewImage( Mat src_image ) {
    return src_image;
  }

  protected Mat cloneImage( Mat src_image ) {
    return src_image.clone();
  }

  protected Mat rotateImage( Mat src_image, double rotation_value ) {
    return src_image;
  }

  protected ArrayList<Mat> getAllExtractedMatrices() {
    return this.extracted_images;
  }

  protected void addMatrixToList( Mat src_image ) {
    this.extracted_images.add( src_image );
  }

  protected Mat getExtractedMatrixFromListByIndex( int index ) {
    return this.extracted_images.get(index);
  }

  protected void addExtractedMatToListByIndex( Mat extracted_image, int index ) { this.extracted_images.add( index, extracted_image ); }

  protected double getAreaSize() {
    return this.areaSize;
  }

  protected void setAreaSize( double area_size ) {
    this.areaSize = area_size;
  }

  protected void setSrcImage( Mat src_image ) {
    this.src_image = src_image;
  }

  protected Mat getSrcImage() { return this.src_image; }

}
