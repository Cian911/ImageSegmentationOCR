package trusthub.factories;

import trusthub.contructors.CarScreenImage;
import trusthub.contructors.TaxDiscImage;
/**
 * Created by Cian on 1/9/2017.
 */
class ImageFactory {

  public Object buildImageTypeObject( String image_type ) {
    if( image_type == null ) {
      return null;
    }
    if( image_type.equalsIgnoreCase("CAR_SCREEN") ) {
      return new CarScreenImage();
    }
    if( image_type.equalsIgnoreCase("TAX_DISC") ) {
      return new TaxDiscImage();
    }

    return null;
  }

}
