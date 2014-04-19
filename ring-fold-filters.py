import json
from pathlib import Path
import argh
import wfc3_utils
import numpy as np
from astropy.table import Table
from astropy.io import fits

root_dir = Path.cwd().parent.parent
data_dir = root_dir/"RingNebula"/"WFC3"/"2013-Geometry"/"Spectra"

fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N", "F645N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

col_names = ["Section", "PA", "x0"] + fns
col_dtypes = ["<U15", int, float] + [float]*len(fns)

WFC3_CONSTANT = 0.29462         # counts cm^2 sr / erg / Ang / pixel
SLIT_WIDTH = 1.9                # arcsec
PIXEL_SIZE = 1.3                # arcsec
ARCSEC_RADIAN = 180.0*3600.0/np.pi 
PIXEL_AREA_SR = SLIT_WIDTH*PIXEL_SIZE / ARCSEC_RADIAN**2

# The line fluxes were in units of erg/s/cm2/AA/fiber, 
# but have already been multiplied by 1e15 
FACTOR = 1.e-15*WFC3_CONSTANT/PIXEL_AREA_SR

PAs = [60, 150]

def main():
    """Construct a table of per-filter fluxes from the Ring spectra. 

Uses the data written to spectral_fit_fine_db"""

    # Load the spectra
    hdu = {
        150: fits.open(str(data_dir/"sb150ed.fits"))[0],
        60: fits.open(str(data_dir/"sb60ed.fits"))[0]
    }

    # This is the conversion factor between the short-slit sb150ed files
    # and the full-slit E1200PA150 files.  I still need to track down the
    # provenance of this
    conversion = 10**(-5.3)
    # Note that this puts the units to be the same as my previous edited
    # full-slit files, which had been multiplied by 1e15 so that the
    # numbers are of order unity
    snorm = 1./(conversion)
    for pa in PAs:
        hdu[pa].data *= snorm

    # Set up wavelength coordinates - Angstrom
    nx, wav0, i0, dwav = [hdu[150].header[k] for k in 
                          ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
    wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav 

    # Set up position coordinates - pixels from the central star
    ny, x0, j0, dx = [hdu[150].header[k] for k in 
                      ("NAXIS2", "CRVAL2", "CRPIX2", "CD2_2")]
    jshift = 163 # position of star
    xpos = x0 + (np.arange(ny) - (j0 + jshift - 1))*dx 

    # Set up the filter transmission curves
    Tfilters = {fn: wfc3_utils.get_interpolated_filter(fn, wavs) for fn in fns}

    # Get the positions of each slit section
    with open(str(data_dir / "spectral_fit_fine_db.json")) as f:
        sections = json.load(f)

    tab = Table(names=col_names, dtype=col_dtypes)

    for label, section in sections.items():
        table_row = {"Section": label, "PA": section["PA"],
                     "x0": 0.5*(section["x1"] + section["x2"])}
        j1 = np.argmin(np.abs(xpos-section["x1"]))
        j2 = np.argmin(np.abs(xpos-section["x2"])) + 1
        spectrum = hdu[section["PA"]].data[j1:j2, :].mean(axis=0)
        for fn in fns:
            integrand = Tfilters[fn]*spectrum*wavs
            table_row[fn] =  FACTOR*np.trapz(integrand, wavs)
        tab.add_row(table_row)
    tab.sort(['PA', 'x0'])
    tab.write('ring-filter-predicted-rates.tab', format='ascii.tab')


        

if __name__ == "__main__":
    argh.dispatch_command(main)
