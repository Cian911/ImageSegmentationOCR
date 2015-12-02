package trusthub;

import org.opencv.core.*;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;

import java.util.*;


/**
 * Created by Cian on 11/9/2015.
 */
class ImageManipulation {

    private static FeatureDetector fd = FeatureDetector.create(FeatureDetector.MSER);
    private static MatOfKeyPoint mokp = new MatOfKeyPoint();
    private static Mat edges = new Mat();
    private ArrayList<Mat> matList;
    private ArrayList<Mat> cleanedMatList;

    public Mat test (Mat srcImage) {
        Mat mRgba = srcImage.clone();
        Mat mGray = new Mat();

        if (mRgba.channels() > 1) {
            Imgproc.cvtColor(srcImage, mGray, Imgproc.COLOR_RGB2GRAY);
        } else {
            mGray = srcImage.clone();
        }

        Imgproc.Canny(mGray, mGray, 10, 10, 3, true);


        MatOfKeyPoint keypoint = new MatOfKeyPoint();
        List<KeyPoint> listpoint = new ArrayList<KeyPoint>();
        KeyPoint kpoint = new KeyPoint();
        Mat mask = Mat.zeros(mGray.size(), CvType.CV_8UC1);
        int rectanx1;
        int rectany1;
        int rectanx2;
        int rectany2;
        double largestArea = 750000.0;

        //
        Scalar zeos = new Scalar(0, 0, 0);
        List<MatOfPoint> contour1 = new ArrayList<MatOfPoint>();
        List<MatOfPoint> contour2 = new ArrayList<MatOfPoint>();
        Mat kernel = new Mat(1, 50, CvType.CV_8UC1, Scalar.all(255));
        Mat morbyte = new Mat();
        Mat hierarchy = new Mat();

        Rect rectan2 = new Rect();//
        Rect rectan3 = new Rect();//
        int imgsize = mRgba.height() * mRgba.width();
        //
            FeatureDetector detector = FeatureDetector
                    .create(FeatureDetector.MSER);
            detector.detect(mGray, keypoint);
            Features2d.drawKeypoints(mGray, keypoint, mGray);
            listpoint = keypoint.toList();
            //
            for (int ind = 0; ind < listpoint.size(); ind++) {
                kpoint = listpoint.get(ind);
                rectanx1 = (int) (kpoint.pt.x - 0.5 * kpoint.size);
                rectany1 = (int) (kpoint.pt.y - 0.5 * kpoint.size);
                rectanx2 = (int) (kpoint.size);
                rectany2 = (int) (kpoint.size);
                if (rectanx1 <= 0)
                    rectanx1 = 1;
                if (rectany1 <= 0)
                    rectany1 = 1;
                if ((rectanx1 + rectanx2) > mGray.width())
                    rectanx2 = mGray.width() - rectanx1;
                if ((rectany1 + rectany2) > mGray.height())
                    rectany2 = mGray.height() - rectany1;
                Rect rectant = new Rect(rectanx1, rectany1, rectanx2, rectany2);
                Mat roi = new Mat(mask, rectant);
                roi.setTo(new Scalar(255));

            }

            Imgproc.morphologyEx(mask, morbyte, Imgproc.MORPH_DILATE, kernel);
            Imgproc.findContours(morbyte, contour2, hierarchy,
                    Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_NONE);

        matList  = new ArrayList<Mat>();

        for (int i = 0; i < contour2.size(); i++) {
            rectan3 = Imgproc.boundingRect(contour2.get(i));

            if (rectan3.area() > 0.5 * imgsize || rectan3.area() < 10000 || rectan3.width / rectan3.height < 3) {
                Mat roi = new Mat(morbyte, rectan3);
                roi.setTo(zeos);
            } else {
                Imgproc.rectangle(mRgba, rectan3.br(), rectan3.tl(), new Scalar(0, 0, 255));

                if (rectan3.area() > largestArea) {
                    System.out.println("Rect: " + rectan3.area());
                    largestArea = rectan3.area();
                    // Add our rectangles to an ArrayList to be processed
                    Rect nRoi = new Rect(rectan3.br(), rectan3.tl());
                    Mat box = mRgba.submat(nRoi);
                    matList.add(0, box);
                }
            }
        }

        return mRgba;
    }

    public Mat deskewImage (Mat srcImage, double angle) {
        Point center = new Point(srcImage.width()/2, srcImage.height()/2);
        Mat rotateImage = Imgproc.getRotationMatrix2D(center, angle, 1.0);

        Size size = new Size(srcImage.width(), srcImage.height());
        Imgproc.warpAffine(srcImage, srcImage, rotateImage, size, Imgproc.INTER_LINEAR + Imgproc.CV_WARP_FILL_OUTLIERS);

        return srcImage;
    }

    public Mat properSkewAngle(Mat srcImage) {
        Mat lines = new Mat();
        Mat ret = srcImage.clone();
        Imgproc.cvtColor(srcImage, srcImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(srcImage, srcImage, 120, 255, Imgproc.THRESH_BINARY);
        Imgproc.HoughLinesP(srcImage, lines, 1, Math.PI / 180, 100, srcImage.width() / 2, 20);

        double angle = 0;
        Size numLines = lines.size();
        Mat disLines = new Mat(srcImage.size(), CvType.CV_8UC1, new Scalar(255, 0, 0));
//        System.out.println ("Area: " + numLines.area() + ", Size: " + numLines);
        for (int i = 0; i < numLines.area(); i++) {
            double[] l0 = lines.get(i, 0);
            double p0 = (Double.isNaN(l0[0])) ? 0 : l0[0];
            double p1 = 0;
            double p2 = 0;
            double p3 = 0;

            Imgproc.rectangle(disLines, new Point(p0, p1), new Point(p2, p3), new Scalar(255, 0, 0));

            angle += Math.atan2(p3 - p1, p2 - p0);
        }

        angle /= numLines.area();

        Mat newSrc = deskewImage(ret, angle);
        return newSrc;
    }

    public Mat cleanImage (Mat srcImage) {
        Core.normalize(srcImage, srcImage, 0, 255, Core.NORM_MINMAX);
        Imgproc.threshold(srcImage, srcImage, 0, 255, Imgproc.THRESH_OTSU);
        Imgproc.erode(srcImage, srcImage, new Mat());
        Imgproc.dilate(srcImage, srcImage, new Mat(), new Point(0, 0), 5);
        Imgproc.morphologyEx(srcImage, srcImage, Imgproc.MORPH_CLOSE, new Mat());
        Imgproc.medianBlur(srcImage, srcImage, 3);
        return srcImage;
    }

    public Mat newCleanImage (Mat srcImage) {
        Mat im = new Mat();
        srcImage.copyTo(im);
        Mat bw = new Mat(im.size(), CvType.CV_8U);
        Imgproc.threshold(im, bw, 0, 255, Imgproc.THRESH_BINARY_INV | Imgproc.THRESH_OTSU);
// take the distance transform
        Mat dist = new Mat(im.size(), CvType.CV_32F);
        Imgproc.distanceTransform(bw, dist, Imgproc.CV_DIST_L2, Imgproc.CV_DIST_MASK_PRECISE);
// threshold the distance transform
        Mat dibw32f = new Mat(im.size(), CvType.CV_32F);
        final double SWTHRESH = 8.0;    // stroke width threshold
        Imgproc.threshold(dist, dibw32f, SWTHRESH/2.0, 255, Imgproc.THRESH_BINARY);
        Mat dibw8u = new Mat(im.size(), CvType.CV_8U);
        dibw32f.convertTo(dibw8u, CvType.CV_8U);

        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
// open to remove connections to stray elements
        Mat cont = new Mat(im.size(), CvType.CV_8U);
        Imgproc.morphologyEx(dibw8u, cont, Imgproc.MORPH_OPEN, kernel);
// find contours and filter based on bounding-box height
        final double HTHRESH = im.rows() * 0.5; // bounding-box height threshold
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        List<Point> digits = new ArrayList<Point>();    // contours of the possible digits
        Imgproc.findContours(cont, contours, new Mat(), Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int i = 0; i < contours.size(); i++)
        {
            if (Imgproc.boundingRect(contours.get(i)).height > HTHRESH)
            {
                // this contour passed the bounding-box height threshold. add it to digits
                digits.addAll(contours.get(i).toList());
            }
        }
// find the convexhull of the digit contours
        MatOfInt digitsHullIdx = new MatOfInt();
        MatOfPoint hullPoints = new MatOfPoint();
        hullPoints.fromList(digits);
        Imgproc.convexHull(hullPoints, digitsHullIdx);
// convert hull index to hull points
        List<Point> digitsHullPointsList = new ArrayList<Point>();
        List<Point> points = hullPoints.toList();
        for (Integer i: digitsHullIdx.toList())
        {
            digitsHullPointsList.add(points.get(i));
        }
        MatOfPoint digitsHullPoints = new MatOfPoint();
        digitsHullPoints.fromList(digitsHullPointsList);
// create the mask for digits
        List<MatOfPoint> digitRegions = new ArrayList<MatOfPoint>();
        digitRegions.add(digitsHullPoints);
        Mat digitsMask = Mat.zeros(im.size(), CvType.CV_8U);
        Imgproc.drawContours(digitsMask, digitRegions, 0, new Scalar(255, 255, 255), -1);
// dilate the mask to capture any info we lost in earlier opening
        Imgproc.morphologyEx(digitsMask, digitsMask, Imgproc.MORPH_DILATE, kernel);
// cleaned image ready for OCR
        Mat cleaned = Mat.zeros(im.size(), CvType.CV_8U);
        dibw8u.copyTo(cleaned, digitsMask);

        return cleaned;
    }

    public Mat customClean (Mat srcImage) {
        Imgproc.cvtColor(srcImage, srcImage, Imgproc.COLOR_BGR2GRAY);
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(3, 3));
        // open to remove connections to stray elements
        Mat cont = new Mat(srcImage.size(), CvType.CV_8U);
        Imgproc.morphologyEx(srcImage, cont, Imgproc.MORPH_OPEN, kernel);

        // Find all contours
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(srcImage, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        for (int i = 0; i < contours.size(); i++) {
            Rect rect = Imgproc.boundingRect(contours.get(i));

            if (rect.width < rect.height) {
                Imgproc.rectangle(srcImage, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(0, 0, 255));
            }
        }

        return srcImage;
    }

    public ArrayList<Mat> getCleanedMatList () {
        return cleanedMatList;
    }

    public ArrayList<Mat> getMatList () {
        return matList;
    }


    public static Mat MSER (Mat srcImage) {
        fd.detect(srcImage, mokp);
        KeyPoint[] refKp = mokp.toArray();
        Point[] refPts = new Point[refKp.length];

        for (int i = 0; i < refKp.length; i++) {
            refPts[i] = refKp[i].pt;
        }
        MatOfPoint2f refMatPt = new MatOfPoint2f(refPts);
        MatOfPoint2f approxCurve = new MatOfPoint2f();

        //Processing on mMOP2f1 which is in type MatOfPoint2f
        double approxDistance = Math.random() * 0.2;
        Imgproc.approxPolyDP(refMatPt, approxCurve, approxDistance, true);

        //Convert back to MatOfPoint
        MatOfPoint points = new MatOfPoint(approxCurve.toArray());
        // Get bounding rect
        Rect rect = Imgproc.boundingRect(points);
        // draw enclosing rectangle (all same color, but you could use variable i to make them unique)
        Imgproc.rectangle(srcImage, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0,0,255), 3);
        Features2d.drawKeypoints(srcImage, mokp, srcImage);
        return srcImage;
    }

    /*
    * t() is the transpose of the matrix
    * */
    public Mat rotateImage (Mat image) {

        Mat outputImg = new Mat();
        // Rotate 90deg clockwise
        Core.flip(image.t(), outputImg, 1);

        return outputImg;
    }

    public double skew (Mat srcImage) {
        Imgproc.threshold(srcImage, srcImage, 200, 255, Imgproc.THRESH_BINARY_INV);

        ArrayList<Point> points = new ArrayList<Point>();

        for (int j = 0; j < srcImage.rows(); j++) {
            for (int i = 0; i < srcImage.cols(); i++) {
                double[] pixel = srcImage.get(j, i);
                if (pixel[0] == 255) {
                    points.add(new Point(j, i));
                }
            }
        }

        MatOfPoint2f newPoints = new MatOfPoint2f();
        newPoints.fromList(points);
        MatOfPoint2f m2f = new MatOfPoint2f();
        newPoints.convertTo(m2f, CvType.CV_32FC2);

        RotatedRect minRect = Imgproc.minAreaRect(newPoints);
        Mat rotateMatrix = Imgproc.getRotationMatrix2D(minRect.center, minRect.angle, 1);
        return minRect.angle;
    }

    public Mat apply(Mat src, Mat dst) {
        if (dst != src) {
            src.copyTo(dst);
        }
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
        double largestArea = 0;

        matList  = new ArrayList<Mat>();
        int c = 0;

        for( int i = 0; i < contours.size(); i++ )

            if (contours.get(i).toList().size()>1200)
            {
                contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);
                Imgproc.approxPolyDP(mMOP2f1,mMOP2f2, 3, true );
                mMOP2f2.convertTo(contours_poly.get(i), CvType.CV_32S);
                Rect appRect=Imgproc.boundingRect(contours_poly.get(i));
                if (appRect.width > appRect.height)
                {
                    Imgproc.rectangle(dst, new Point(appRect.x,appRect.y) ,new Point(appRect.x+appRect.width,appRect.y+appRect.height), new Scalar(255,0,0));

                    if (appRect.area() > largestArea) {
                        largestArea = appRect.area();
                        // Add our rectangles to an ArrayList to be processed
                        Rect nRoi = new Rect(appRect.br(), appRect.tl());
                        Mat box = dst.submat(nRoi);
                        matList.add(c, box);
                    }
                }
            }
        return dst;
    }

    public Mat findCountours (Mat srcImage) {
        List<MatOfPoint> imageCountours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        srcImage.convertTo(srcImage, CvType.CV_32SC1);

        Imgproc.findContours(srcImage, imageCountours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0,0));

        hierarchy.release();

        MatOfPoint2f approxCurve = new MatOfPoint2f();

        for (int i = 0; i < imageCountours.size(); i++) {
            MatOfPoint2f countour2f = new MatOfPoint2f(imageCountours.get(i).toArray());

            double approxDistance = Imgproc.arcLength(countour2f, true) * 0.03;

            Imgproc.approxPolyDP(countour2f, approxCurve, approxDistance, true);

            // Convert back to MatOfPoint
            MatOfPoint points = new MatOfPoint(approxCurve.toArray());

            // Get bounding of countour
            Rect rect = Imgproc.boundingRect(points);
            // Draw rectangles
            Imgproc.rectangle(srcImage, new Point(rect.x,rect.y), new Point(rect.x+rect.width,rect.y+rect.height), new Scalar(255, 0, 0), 3);
        }

        return srcImage;
    }

}
