from astropy.io import fits
import numpy as np
import os
import sys
import json
import coord_utils
import scipy.ndimage as ni
from astropy.table import Table, join

wfc3_pix = 0.04                 # arcsec
# Seeing/focus was determined by fitting Gaussians to the star profiles
seeing_FWHM = 2.63               # arcsec
sig_factor = 1.0/(2.0*np.sqrt(2.0*np.log(2.0)))


datadir = "."

# PA 60 slit is offest slightly from central star
# Central coordinates were determined by manually aligning the slit and comparing profiles
# See WFC3\ Ring\ Comparison.pynb
slit_center = {
    60: "5:35:16.5 -5:23:53",
    90: "5:35:16.5 -5:24:23",
}
slit_width = 1.9                # arcsec
# From Sec 2.1 of ODH 2010
pixel_size = 1.3               # arcsec 
# (taking slit length of 330 pix = 3 x 143 arcsec - from Tab 2 of ODH2010)

# From fitting Gaussians, these are the 0-base pixel positions of the
# central stars along the slit
xzeropoint = {60: 52.95, 150: 52.9}
# Star position I had assumed in ring-photom-fine.py, but with a
# correction by hand since things still were not lining up
j0 = 55 - 1.5


Filters = {
    "FQ436N": "FQ436N-drz-bg.fits",
    "FQ575N": "FQ575N-drz-bg.fits",
    "FQ672N": "FQ672N-drz-bg.fits",
    "FQ674N": "FQ674N-drz-bg.fits",
    "F673N": "F673N-drz-bg.fits",
    "F469N": "F469N-drz-bg.fits",
    "F487N": "F487N-drz-cr.fits",
    "F656N": "F656N-drz-cr.fits",
    "F658N": "F658N-drz-cr.fits",
    "F547M": "F547M-drz-bg.fits",
    "F645N": "F645N-drz-bg.fits",
    "F502N": "F502N-drz-cr.fits",
    "FQ437N": "FQ437N-drz-bg.fits"
}

with open("Spectra/spectral_fit_fine_db.json") as f:
    db = json.load(f)

sections = db.keys()
results = {}
bigtable = None
for Filter, fname in Filters.items():
    hdu = fits.open(os.path.join(datadir, fname))["SCI"]
    RA, DEC = coord_utils.get_radec(hdu.header)
    sdata = ni.gaussian_filter(hdu.data, seeing_FWHM*sig_factor/wfc3_pix, mode='constant')
    sfname = fname.replace("-drz", "-s{:03d}".format(int(100*seeing_FWHM)))
    fits.PrimaryHDU(sdata, hdu.header).writeto(os.path.join(datadir, sfname), clobber=True)
    results[Filter] = {}
    thistable = Table(names=["PA", "Section", "x0", Filter], dtype=[int, '<U6', float, float])
    for PA in 150, 60:
        sections = sorted(section for section in db if db[section]["PA"] == PA)
        xslit, yslit = coord_utils.slitxy_from_radec(RA, DEC, slit_center[PA], PA)
        xslit = -xslit # PAs are actually the wrong way round
        slitmask = np.abs(yslit) <= slit_width/2.0
        for section in sections:
            x1 = pixel_size*(db[section]["x1"] + j0 - xzeropoint[PA]) 
            x2 = pixel_size*(db[section]["x2"] + j0 - xzeropoint[PA])
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



bigtable.write("ring_calibration_db.tab", format="ascii.tab")

