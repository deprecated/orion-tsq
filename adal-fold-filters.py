import argh
import wfc3_utils
import numpy as np
from astropy.table import Table
from astropy.io import fits
import json
import adal_common
from coord_utils import radec_offsets_from_slitx

fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

col_names = ["Section", "x0", "dRA", "dDEC"] + fns
# I hate fixed-width strings!  A constant source of bugs.  I give a
# few extra chars in the first column here to allow for future growth
col_dtypes = ["<U15"] + [float, float, float] + [float]*len(fns)
# Avoid excessive precision in floats in the output table - 4 sig figs
# is plenty
col_fmts = {fn: '%.4g' for fn in fns}
col_fmts.update({k: '%.2f' for k in ['x0', 'dRA', 'dDEC']})

bands = ["red", "blue"]
slits = [5, 6]

WFC3_CONSTANT = 0.29462         # counts cm^2 sr / erg / Ang / WFC3 pixel
SLIT_WIDTH = adal_common.slit_width              # arcsec
PIXEL_SIZE = adal_common.pixel_size              # arcsec
ARCSEC_RADIAN = 180.0*3600.0/np.pi 
# The line fluxes were in units of erg/s/cm2/AA/pixel, 
# This FACTOR should convert to counts per second per WFC3 pixel
FACTOR = WFC3_CONSTANT*ARCSEC_RADIAN**2/(SLIT_WIDTH*PIXEL_SIZE)


def main():
    """Construct a table of per-filter fluxes from the ADAL spectra. 

"""

    hdu = {
        ('red',  5): fits.open("Adal-Slits/zorip5rojo_1d.fits")[1],
        ('red',  6): fits.open("Adal-Slits/zorip6rojo_1d.fits")[1],
        ('blue',  5): fits.open("Adal-Slits/zorip5azul_1d.fits")[1],
        ('blue',  6): fits.open("Adal-Slits/zorip6azul_1d.fits")[1],
    }
    
    # Set up table to hold the results 
    tab = Table(names=col_names, dtype=col_dtypes)

    # Set up slit sections
    sections = {}

    for band in bands:

        # Set up wavelength coordinates - Angstrom
        # This is the same for each slit
        nx, wav0, i0, dwav = [hdu[(band, 5)].header[k] for k in 
                              ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
        # Subtract 1 from i0 because it is 1-based 
        wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav 
        
        # Set up the filter transmission curves
        Tfilters = {fn: wfc3_utils.get_interpolated_filter(fn, wavs) for fn in fns}

        for islit in slits:
            # # Set up position coordinates along slit - arcsec from slit center
            ny = hdu[(band, islit)].header['NAXIS2']
            j0 = ny/2 + 0.5
            x0 = adal_common.slit_xshift[islit]
            dx = PIXEL_SIZE
            # Slits are upside-down wrt the stated PA, hence the minus sign
            xslit = -(x0 + (np.arange(ny) - j0)*dx)
            dRA, dDEC = radec_offsets_from_slitx(xslit,
                                                 center=adal_common.slit_center[islit],
                                                 PA=adal_common.slit_PA[islit])
            for j in range(ny):
                key = "S{:01d}-{:s}-{:03d}".format(islit, band, j)
                sections[key] = {"Slit": islit, "band": band, "j": j, "x": xslit[j]}
                table_row = {"Section": key, "x0": xslit[j],
                             "dRA": dRA[j], "dDEC": dDEC[j]}
                spectrum = hdu[(band, islit)].data[j, :]
                for fn in fns:
                    if band == adal_common.Bands[fn]:
                        integrand = Tfilters[fn]*spectrum*wavs
                        table_row[fn] =  FACTOR*np.trapz(integrand, wavs)
                tab.add_row(table_row)


    # Save these slit sections, which might be different from the ones
    # used in the line fitting
    with open('adal-fold-sections.json', 'w') as f:
        json.dump(sections, f, indent=2)

    # And save the filter intensities
    tab.write('adal-filter-predicted-rates.tab', format='ascii.tab', formats=col_fmts)


        

if __name__ == "__main__":
    argh.dispatch_command(main)
