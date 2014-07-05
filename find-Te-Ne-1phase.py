
import pyneb
from scipy.interpolate import griddata
from astropy.io import fits
import numpy as np

pyneb.atomicData.resetDataFileDict()
sii = pyneb.Atom("S", 2)
nii = pyneb.Atom("N", 2)

def rsii_func(T, den):
    A = sii.getEmissivity(T, den, wave=6716)
    B = sii.getEmissivity(T, den, wave=6731)
    return A/B

def rnii_func(T, den):
    A = nii.getEmissivity(T, den, wave=5755)
    B = nii.getEmissivity(T, den, wave=6583)
    return A/B

Tmin, Tmax = 2000.0, 20000.0
Nmin, Nmax = 100.0, 5.e4
nT, nN = 400, 400
Tgrid = np.linspace(Tmin, Tmax, nT)
Ngrid = np.logspace(np.log10(Nmin), np.log10(Nmax), nN)
Nvalues, Tvalues = np.meshgrid(Ngrid, Tgrid)

rsii_grid = rsii_func(Tgrid, Ngrid)
rnii_grid = rnii_func(Tgrid, Ngrid)

hdu_sii = fits.open("ratio-FQ672N-FQ674N-masked.fits")[0]
hdu_nii = fits.open("ratio-FQ575N-F658N-masked.fits")[0]
rsii = hdu_sii.data
rnii = hdu_nii.data
m = np.isfinite(rsii) & np.isfinite(rnii)

# Deal with rsii values outside theoretical range
rsii[rsii < rsii_grid.min()]= rsii_grid.min()
rsii[rsii > rsii_grid.max()]= rsii_grid.max()

xi = np.array(zip(rsii[m], rnii[m]))
points = np.array(zip(rsii_grid.ravel(), rnii_grid.ravel()))

Te = griddata(points, Tvalues.ravel(), xi, method='nearest')
Ne = griddata(points, Nvalues.ravel(), xi, method='nearest')

hdu_sii.data[m] = Ne
hdu_sii.data[~m] = np.nan
hdu_nii.data[m] = Te
hdu_nii.data[~m] = np.nan

hdu_nii.writeto('Te-1phase-pyneb.fits', clobber=True)
hdu_sii.writeto('Ne-1phase-pyneb.fits', clobber=True)
