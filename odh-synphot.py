#!/Users/will/anaconda/envs/py27/bin/python
import argh
import wfc3_utils
import numpy as np
from astropy.table import Table
from astropy.io import fits
import pysynphot                # REQUIRES python 2.7
import json

fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N", "F645N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

col_names = ["Section"] + fns
col_dtypes = ["<U8"] + [float]*len(fns)

bands = ["red", "green", "blue"]
wavcache = {}
Tcache = {"red": {}, "green": {}, "blue": {}}

SLIT_WIDTH = 1.9                # arcsec
PIXEL_SIZE = 1.3                # arcsec

# Convert between spectrograph aperture size and WFC3 pixel size
FACTOR = (0.04**2)/(SLIT_WIDTH*PIXEL_SIZE)


def main(seclength=2):
    """Construct a table of per-filter fluxes from the ODH spectra. 

Re-written to use pysynphot instead of doing things myself"""

    hdu = {
        30: fits.open("ODell-Harris/S30.30ed.fits")[0],
        60: fits.open("ODell-Harris/S60.30.fits")[0],
        90: fits.open("ODell-Harris/S90.60ed.fits")[0]
    }
    Offsets = list(hdu.keys())

    # Do not multiply by 1e15 this time

    # Set up wavelength coordinates - Angstrom
    nx, wav0, i0, dwav = [hdu[60].header[k] for k in 
                          ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
    wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav 
    vacwavs = wavs*wfc3_utils.AIR_REFRACTIVE_INDEX

    # # Set up position coordinates - pixels from the central star
    ny, x0, j0, dx = [hdu[60].header[k] for k in 
                      ("NAXIS2", "CRVAL2", "CRPIX2", "CD2_2")]
    # xpos = x0 + (np.arange(ny) - (j0 - 1))*dx 

    # Set up the filter transmission curves
    Tfilters = {fn: pysynphot.ObsBandpass('wfc3,uvis1,' + fn) for fn in fns}
        
    # Set up table to hold the results 
    tab = Table(names=col_names, dtype=col_dtypes)

    # Set up slit sections
    # Much simpler than the for the Ring Nebula - just use integer blocks
    sections = {}
    nsections = ny//seclength
    for offset in Offsets:
        for n in range(nsections):
            key = "S{}-{:03d}".format(offset, n)
            sections[key] = {"Offset": offset,
                             "j1": n*seclength,
                             "j2": (n+1)*seclength}

    # Save these slit sections, which might be different from the ones
    # used in the line fitting
    with open('odh-fold-sections.json', 'w') as f:
        json.dump(sections, f)

    for label, section in sections.items():
        table_row = {"Section": label}
        offset = section["Offset"]
        j1 = section["j1"]
        j2 = section["j2"]
        flux_per_pixel = FACTOR*hdu[offset].data[j1:j2, :].mean(axis=0)
        spectrum = pysynphot.ArraySpectrum(wave=vacwavs, flux=flux_per_pixel,
                                           fluxunits='flam')
        for fn in fns:
            bandpass = Tfilters[fn]
            observation = pysynphot.Observation(spectrum, bandpass,
                                                force='extrapolate')
            table_row[fn] = observation.countrate()
        tab.add_row(table_row)
    tab.write('odh-filter-synphot-rates.tab', format='ascii.tab')
        

if __name__ == "__main__":
    argh.dispatch_command(main)
