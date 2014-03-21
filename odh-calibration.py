from astropy.io import fits
import numpy as np
import os
import json
import coord_utils
import scipy.ndimage as ni
from astropy.table import Table, join

wfc3_pix = 0.04                 # arcsec
# Seeing/focus was determined by fitting Gaussians to the star profiles
seeing_FWHM = 1.5               # arcsec
sig_factor = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))


datadir = "."

# Central coordinates were taken from ODH paper
# Except S90 is adjusted to have same RA as the others
slit_center = {
    "S30": "5:35:16.5 -5:23:53",
    "S60": "5:35:16.5 -5:24:23",
    "S90": "5:35:16.5 -5:24:53",
}
slit_ids = sorted(list(slit_center))
slit_width = 1.9                # arcsec
# From Sec 2.1 of ODH 2010
pixel_size = 1.3               # arcsec 
# (taking slit length of 330 pix = 3 x 143 arcsec - from Tab 2 of ODH2010)
PA = -90                         # All slits are E-W
# Center of slit in zero-based pixel units
ny = 330
j0 = 0.5*(ny - 1)

Filters = {
    "FQ436N": "full_FQ436N_north_pad.fits",
    "FQ575N": "full_FQ575N_north_pad.fits",
    "FQ672N": "full_FQ672N_north_pad.fits",
    "FQ674N": "full_FQ674N_north_pad.fits",
    "F673N": "full_F673N_north_pad.fits",
    "F469N": "full_F469N_north_pad.fits",
    "F487N": "full_F487N_north_pad.fits",
    "F656N": "full_F656N_north_pad.fits",
    "F658N": "full_F658N_north_pad.fits",
    "F547M": "full_F547M_north_pad.fits",
#    "F645N": "F645N_north_pad.fits",
    "F502N": "full_F502N_north_pad.fits",
    "FQ437N": "full_FQ437N_north_pad.fits"
}

with open("odh_spectral_fit_db.json") as f:
    db = json.load(f)

sections = db.keys()
results = {}
bigtable = None
for Filter, fname in Filters.items():
    hdu = fits.open(os.path.join(datadir, fname))[0]
    RA, DEC = coord_utils.get_radec(hdu.header)
    sdata = ni.gaussian_filter(hdu.data, seeing_FWHM*sig_factor/wfc3_pix, mode='constant')
    sfname = fname.replace("_north_pad", "-s{:03d}".format(int(100*seeing_FWHM)))
    fits.PrimaryHDU(sdata, hdu.header).writeto(os.path.join(datadir, sfname), clobber=True)
    results[Filter] = {}
    thistable = Table(names=["PA", "Section", "x0", Filter], dtype=[int, '<U6', float, float])
    for slit_id in slit_ids:
        sections = sorted(section for section in db if section.startswith(slit_id))
        xslit, yslit = coord_utils.slitxy_from_radec(RA, DEC, slit_center[slit_id], PA)
        slitmask = np.abs(yslit) <= slit_width/2.0
        for section in sections:
            x1 = pixel_size*(db[section]["j1"] - j0) 
            x2 = pixel_size*(db[section]["j2"] - j0)
            dx = np.abs(x2 - x1)
            x0 = 0.5*(x1 + x2)
            secmask = slitmask & (np.abs(xslit - x0) <= dx/2.0)
            if np.any(secmask):
                flux = sdata[secmask].mean()
                thistable.add_row([int(PA), str(section), float(x0), float(flux)])

    if bigtable is None:
        bigtable = thistable
    else:
        bigtable = join(bigtable, thistable, join_type="outer")



bigtable.write("odh_calibration_db.tab", format="ascii.tab")

