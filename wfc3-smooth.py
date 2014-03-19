from astropy.io import fits
import scipy.ndimage as ni
from astropy import wcs
import numpy as np

FWHM = 1.2                      # arcsec
sig_factor = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))

filters = ["F658N", "F547M", "FQ575N"]
for filter_ in filters:
    whdu = fits.open("full_{}_north_pad.fits".format(filter_))[0]
    wny, wnx = whdu.data.shape
    w = wcs.WCS(whdu.header)
    
    pixscale = whdu.header["CDELT2"]*3600
    sigma = sig_factor*FWHM/pixscale

    clean_data = whdu.data[:, :]
    mask = np.isfinite(clean_data)
    clean_data[~mask] = np.median(clean_data[mask])
    sdata = ni.gaussian_filter(clean_data, sigma)
    sdata[~mask] = np.nan
    whdu.data = sdata
    whdu.writeto("full_{}_smooth{:02d}.fits".format(filter_, int(10*FWHM)),
                 clobber=True)
