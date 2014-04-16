import argh
import wfc3_utils
import numpy as np
from astropy.table import Table
from astropy.io import fits

fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N", "F645N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

col_names = ["Section"] + fns
col_dtypes = ["<U6"] + [float]*len(fns)

bands = ["red", "green", "blue"]
wavcache = {}
Tcache = {"red": {}, "green": {}, "blue": {}}

def main():
    """Construct a table of per-filter fluxes from the ODH spectra. 

We need to repeat some portion of what was in odh-photom.py"""

    hdu = {
        30: fits.open("ODell-Harris/S30.30ed.fits")[0],
        60: fits.open("ODell-Harris/S60.30.fits")[0],
        90: fits.open("ODell-Harris/S90.60ed.fits")[0]
    }
    Offsets = list(hdu.keys())

    # Multiply by 1e15 as with the Helix to make the spectra of order
    # unity for weak lines
    for offset in Offsets:
        hdu[offset].data *= 1.e15

    # Set up wavelength coordinates - Angstrom
    nx, wav0, i0, dwav = [hdu[60].header[k] for k in 
                          ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
    wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav 

    # # Set up position coordinates - pixels from the central star
    ny, x0, j0, dx = [hdu[60].header[k] for k in 
                      ("NAXIS2", "CRVAL2", "CRPIX2", "CD2_2")]
    # xpos = x0 + (np.arange(ny) - (j0 - 1))*dx 

    # Set up the filter transmission curves
    Tfilters = {fn: wfc3_utils.get_interpolated_filter(fn, wavs) for fn in fns}
        
    # Set up table to hold the results 
    tab = Table(names=col_names, dtype=col_dtypes)

    # Set up slit sections
    # Much simpler than the for the Ring Nebula - just use integer blocks
    seclength = 10 # in pixels
    sections = {}
    nsections = ny//seclength
    for offset in Offsets:
        for n in range(nsections):
            key = "S{}-{:02d}".format(offset, n)
            sections[key] = {"Offset": offset,
                             "j1": n*seclength,
                             "j2": (n+1)*seclength}

    for label, section in sections.items():
        table_row = {"Section": label}
        offset = section["Offset"]
        j1 = section["j1"]
        j2 = section["j2"]
        spectrum = hdu[offset].data[j1:j2, :].mean(axis=0)
        for fn in fns:
            table_row[fn] =  np.trapz(Tfilters[fn]*spectrum, wavs)
        tab.add_row(table_row)
    tab.write('odh-filter-fluxes.tab', format='ascii.tab')


        

if __name__ == "__main__":
    argh.dispatch_command(main)
