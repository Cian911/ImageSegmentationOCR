package trusthub.exceptions;

/**
 * Created by Cian on 11/28/2015.
 */
public class ImageExceptions extends Exception {

    public ImageExceptions (String message) {
        super(message);
    }

    public ImageExceptions (Throwable cause) {
        super(cause);
    }

    public ImageExceptions (String message, Throwable cause) {
        super(message, cause);
    }

}
