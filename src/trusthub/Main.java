package trusthub;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.math.BigInteger;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.List;

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
        Mat outImage = im.rotateImage(srcImage);
        Mat deskew = im.deskewImage(outImage, 90);
        Mat canny = im.test(deskew);

        ArrayList<Mat> matList;
        matList = im.getMatList();
        System.out.println("Size: " + matList.size());
        for (int i = 0; i < matList.size(); i++) {
            im.properSkewAngle(matList.get(i));
            String rand = randomString();
            //matList.set(i, im.deNoise(matList.get(i)));
            matList.set(i, im.cleanImage(matList.get(i)));
            Imgcodecs.imwrite("src/assets/extract/"+rand+".png", matList.get(i));
        }
        Write writeImage = new Write(canny, "src/assets/taxdisc3.png");
        writeImage.writeImageToLocation();
    }

    public static String randomString () {
        SecureRandom random = new SecureRandom();
        return new BigInteger(52, random).toString(32);
    }
}