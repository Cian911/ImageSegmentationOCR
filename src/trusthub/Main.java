package trusthub;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import trusthub.exceptions.ImageExceptions;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.ArrayList;

/**
 * Created by Cian on 11/8/2015.
 */
class Main {
    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        Read readImage = new Read();
        readImage.readImageFromSrc();
        Mat srcImage = readImage.returnSrcImage();

        ImageManipulation im = new ImageManipulation();
//        Mat outImage = im.rotateImage(srcImage);
        Mat outImage = srcImage;
        Mat deskew = im.deskewImage(outImage, 0);
        Mat canny = im.segmentImage(deskew);

        ArrayList<Mat> matList;
        matList = im.getMatList();

        try {

            if (matList.size() == 0) {
                throw new ImageExceptions("Image Text Not detected");
            }

        } catch (ImageExceptions ex) {
            System.err.println(ex);
        }

        for (int i = 0; i < matList.size(); i++) {
            im.properSkewAngle(matList.get(i));
            String rand = randomString();
            matList.set(i, im.cleanImage(matList.get(i)));
            Imgcodecs.imwrite("src/assets/extract/" + rand + ".png", matList.get(i));
        }

        for (int i = 0; i < matList.size(); i++) {
            String rand = randomString();
            matList.set(i, im.customCleanImage(matList.get(i)));

            Imgcodecs.imwrite("src/assets/extract/newtest/" + rand + ".png", matList.get(i));

            String s = null;

            // Execute tesseract
            try {
                Process p = Runtime.getRuntime().exec("tesseract -psm 7 src/assets/extract/newtest/" + rand + ".png ocr/out-" + rand + " digits");
                p.waitFor();

                BufferedReader stdInput = new BufferedReader(new
                        InputStreamReader(p.getInputStream()));

                BufferedReader stdError = new BufferedReader(new
                        InputStreamReader(p.getErrorStream()));

                // read the output from the command
                System.out.println("Here is the standard output of the command:\n");
                while ((s = stdInput.readLine()) != null) {
                    System.out.println(s);
                }

                // read any errors from the attempted command
                System.out.println("Here is the standard error of the command (if any):\n");
                while ((s = stdError.readLine()) != null) {
                    System.out.println(s);
                }
            } catch (IOException | InterruptedException e) {
                System.out.println(e);
            }
        }

        Write writeImage = new Write(canny, "src/assets/taxdisc3.png");
        writeImage.writeImageToLocation();
    }

    public static String randomString() {
        SecureRandom random = new SecureRandom();
        return new BigInteger(52, random).toString(32);
    }
}
