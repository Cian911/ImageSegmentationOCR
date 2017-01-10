package trusthub;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Cian on 1/27/2016.
 */
class CarScreenDetection {

    private List<MatOfPoint> contours;
    private List<MatOfPoint> squares;

    public double angle (Point pt1, Point pt2, Point pt3) {
        // Calculate Distance transform
        double dx1 = pt1.x - pt3.x;
        double dy1 = pt1.y - pt3.y;
        double dx2 = pt2.x - pt3.x;
        double dy2 = pt2.y - pt3.y;

        return (dx1 * dx2 + dy1 * dy2)
                / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
    }

    public void findSquaresInImage (Mat srcImage) {

        Mat pyrUp = new Mat();
        Mat pyrDown = new Mat();

        // Filter out small noises in the image
        Imgproc.pyrUp(srcImage, pyrUp, new Size(srcImage.cols() / 2, srcImage.rows() / 2));
        Imgproc.pyrDown(pyrUp, pyrDown, srcImage.size());

        // Apply edge detection
        Imgproc.Canny(pyrDown, pyrDown, 0, 50, 5, true);

        // Dilate to remove potential holes between edge segments
        Mat kernel = new Mat(-1, -1, CvType.CV_8UC1, Scalar.all(255));
        Imgproc.dilate(pyrDown, pyrDown, kernel);

        // Find contours and store them as a list
        Mat hierarchy = new Mat();
        Imgproc.findContours(pyrDown, this.contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        for ( int i = 0; i < contours.size(); i++ ) {
            MatOfPoint2f approxPoints = new MatOfPoint2f();
            contours.toArray();
            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i)), approxPoints, Imgproc.arcLength(new MatOfPoint2f(contours.get(i)), true) * 0.02, true);

//            if ( approxPoints.size() == 4) {
//
//            }
        }
    }

    public void findSquaresTwo(Mat image, List<MatOfPoint> squares) {
        squares.clear();

        Mat smallerImg=new Mat(new Size(image.width()/2, image.height()/2),image.type());

        Mat gray=new Mat(image.size(),image.type());

        Mat gray0=new Mat(image.size(),CvType.CV_8U);

        // down-scale and upscale the image to filter out the noise
        Imgproc.pyrDown(image, smallerImg, smallerImg.size());
        Imgproc.pyrUp(smallerImg, image, image.size());

        int N = 5;

        // find squares in every color plane of the image
        for( int c = 0; c < 3; c++ )
        {

            Core.extractChannel(image, gray, c);

            // try several threshold levels
            for( int l = 1; l < N; l++ )
            {
                //Cany removed... Didn't work so well


                Imgproc.threshold(gray, gray0, (l+1)*255/N, 255, Imgproc.THRESH_BINARY);


                List<MatOfPoint> contours=new ArrayList<MatOfPoint>();

                // find contours and store them all as a list
                Imgproc.findContours(gray0, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

                MatOfPoint2f approx = new MatOfPoint2f();
                MatOfPoint approxConvert = new MatOfPoint();
                List<MatOfPoint> contourMap = new ArrayList<MatOfPoint>();

                // test each contour
                for( int i = 0; i < contours.size(); i++ )
                {

                    // approximate contour with accuracy proportional
                    // to the contour perimeter
                    Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), approx, Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()), true) * 0.02, true);

                    // square contours should have 4 vertices after approximation
                    // relatively large area (to filter out noisy contours)
                    // and be convex.
                    // Note: absolute value of an area is used because
                    // area may be positive or negative - in accordance with the
                    // contour orientation
                    approx.convertTo(approxConvert, CvType.CV_32S);
                    contourMap.add(approxConvert);

                    if( approx.toArray().length == 4 &&
                            Math.abs(Imgproc.contourArea(approx)) > 1000 &&
                            Imgproc.isContourConvex(approxConvert) )
                    {
                        double maxCosine = 0;

                        for( int j = 2; j < 5; j++ )
                        {
                            // find the maximum cosine of the angle between joint edges
                            double cosine = Math.abs(angle(approx.toArray()[j%4], approx.toArray()[j-2], approx.toArray()[j-1]));
                            maxCosine = Math.max(maxCosine, cosine);
                        }

                        // if cosines of all angles are small
                        // (all angles are ~90 degree) then write quandrange
                        // vertices to resultant sequence
                        if( maxCosine < 0.3 )
                            squares.add(approxConvert);
                    }
                }
            }
        }
    }

    public Mat quickTest(Mat src) {
        Mat dst = new Mat();
        Mat img_gray,img_sobel, img_threshold, element;

        img_gray=new Mat();
        Imgproc.cvtColor(src, img_gray, Imgproc.COLOR_RGB2GRAY);

        img_sobel=new Mat();
        Imgproc.Sobel(img_gray, img_sobel, CvType.CV_8U, 1, 0, 3, 1, 0,Core.BORDER_DEFAULT);

        img_threshold=new Mat();
        Imgproc.threshold(img_sobel, img_threshold, 0, 255, Imgproc.THRESH_OTSU+Imgproc.THRESH_BINARY);

        element=new Mat();
        element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(17, 3) );
        Imgproc.morphologyEx(img_threshold, img_threshold, Imgproc.MORPH_CLOSE, element);
        //Does the trick
        List<MatOfPoint>  contours=new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(img_threshold, contours, hierarchy, 0, 1);
        List<MatOfPoint> contours_poly=new ArrayList<MatOfPoint>(contours.size());
        contours_poly.addAll(contours);

        MatOfPoint2f mMOP2f1,mMOP2f2;
        mMOP2f1=new MatOfPoint2f();
        mMOP2f2=new MatOfPoint2f();

        for( int i = 0; i < contours.size(); i++ )

            if (contours.get(i).toList().size()>100)
            {
                contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
                Imgproc.approxPolyDP(mMOP2f1,mMOP2f2, 3, true );
                mMOP2f2.convertTo(contours_poly.get(i), CvType.CV_32S);
                Rect appRect=Imgproc.boundingRect(contours_poly.get(i));
                if (appRect.width>appRect.height)
                {
                    Imgproc.rectangle(dst, new Point(appRect.x,appRect.y) ,new Point(appRect.x+appRect.width,appRect.y+appRect.height), new Scalar(255,0,0));
                }
            }
        return dst;
    }

}
