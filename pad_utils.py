
import numpy as np
from astropy.io import fits
nxx, nyy = 2999, 2999
ii0, jj0 = 1500.0, 1500.0

def pad_image_to_header(hdu):
    """Pad to a common size with alignment of the reference pixel
    """
    outimage = np.empty((nyy, nxx), dtype=float)
    outimage[:,:] = np.nan
    i0, j0 = hdu.header["CRPIX1"], hdu.header["CRPIX2"]  
    nx, ny = hdu.header["NAXIS1"], hdu.header["NAXIS2"]  
    # Corners of output image to fill with input image
    ii1 = ii0 - i0
    ii2 = ii1 + nx
    jj1 = jj0 - j0
    jj2 = jj1 + ny
    # Fill it in
    inshape = hdu.data.shape
    outshape = outimage[jj1:jj2, ii1:ii2].shape
    assert outshape == inshape, (ii1, ii2, jj1, jj2, outshape, inshape)
    outimage[jj1:jj2, ii1:ii2] = hdu.data
    return outimage

def pad(filt):
    prefix = "full_" + filt
    hdu = fits.open(prefix + "_north.fits")[0]
    hdu.data = pad_image_to_header(hdu)
    hdu.header["NAXIS1"] = nxx
    hdu.header["NAXIS2"] = nyy
    hdu.header["CRPIX1"] = ii0
    hdu.header["CRPIX2"] = jj0
    hdu.writeto(prefix + "_north_pad.fits", clobber=True)
