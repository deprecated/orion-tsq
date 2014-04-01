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

fiber_radius = 2.69/2.0         # arcsec 

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

with open("manu_spectral_fit_db.json") as f:
    db = json.load(f)
positions = sorted(db.keys())

PA = 90
th1C_coords = "05:35:16.463 -5:23:23.18"
results = {}
bigtable = None
for Filter, fname in Filters.items():
    print("Extracting aperture fluxes from", fname)
    hdu = fits.open(os.path.join(datadir, fname))[0]
    RA, DEC = coord_utils.get_radec(hdu.header)
    sdata = ni.gaussian_filter(hdu.data, seeing_FWHM*sig_factor/wfc3_pix, mode='constant')
    sfname = fname.replace("_north_pad", "-s{:03d}".format(int(100*seeing_FWHM)))
    fits.PrimaryHDU(sdata, hdu.header).writeto(os.path.join(datadir, sfname), clobber=True)
    results[Filter] = {}
    thistable = Table(names=["x", "y", "Section", Filter], dtype=[float, float, '<U15', float])
    xim, yim = coord_utils.slitxy_from_radec(RA, DEC, th1C_coords, PA)
    for position in positions:
        x0 = db[position]['x']
        y0 = db[position]['y']
        posmask = np.hypot(xim - x0, yim - y0) <= fiber_radius
        if np.any(posmask):
            flux = sdata[posmask].mean()
            thistable.add_row([float(x0), float(y0), str(position), float(flux)])

    if bigtable is None:
        bigtable = thistable
    else:
        bigtable = join(bigtable, thistable, join_type="outer")



bigtable.write("manu_calibration_db.tab", format="ascii.tab")

