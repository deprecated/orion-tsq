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

SLIT_WIDTH = adal_common.slit_width              # arcsec
PIXEL_SIZE = adal_common.pixel_size              # arcsec
# The line fluxes were in units of erg/s/cm2/AA/pixel, 
# This FACTOR should convert to counts per second per WFC3 pixel
FACTOR = wfc3_utils.WFC3_CONSTANT*wfc3_utils.ARCSEC_RADIAN**2/(SLIT_WIDTH*PIXEL_SIZE)


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

    # Correct zero point of blue spectra. This is completely ad hoc,
    # but it is necessary to make the calibration plots go through the
    # origin
    for islit in slits:
        hdu[('blue', islit)].data += 7.0e-16

    for band in bands:

        # Set up wavelength coordinates - Angstrom
        # This is the same for each slit
        nx, wav0, i0, dwav = [hdu[(band, 5)].header[k] for k in 
                              ("NAXIS1", "CRVAL1", "CRPIX1", "CD1_1")]
        # Subtract 1 from i0 because it is 1-based 
        wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav 

        # extend and extrapolate red spectrum to cover the entire F547M bandpass
        if band == 'red':
            dwav, = np.diff(wavs[:2])
            i1 = np.argmin(np.abs(wavs - 5450.0))
            i2 = np.argmin(np.abs(wavs - 5700.0))
            nx_extend = int((wavs[i1] - 4900)/dwav)
            wavs_extend = wavs[i1] - (1. + np.arange(nx_extend)[::-1])*dwav
            wavs = np.hstack((wavs_extend, wavs[i1:]))
            for islit in slits:
                fill_values = np.mean(hdu[('red', islit)].data[:, i1:i2], axis=1)
                data_extend = np.ones(nx_extend)[None, :] * fill_values[:, None]
                hdu[('red', islit)].data = np.hstack((data_extend,
                                                      hdu[('red', islit)].data[:, i1:]))
        # Vacuum wavelengths are necessary  to use with the filter curves
        vacwavs = wavs*wfc3_utils.AIR_REFRACTIVE_INDEX
        
        # Set up the filter transmission curves
        Tfilters = {fn: wfc3_utils.get_interpolated_filter(fn, vacwavs) for fn in fns}

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
                        if fn == 'F502N':
                            # Spot correction for the 5007 line, based
                            # on forcing the the average 5007/4959
                            # ratio to be the theoretical value of 2.918
                            table_row[fn] *= 1.62
                tab.add_row(table_row)


    # Save these slit sections, which might be different from the ones
    # used in the line fitting
    with open('adal-fold-sections.json', 'w') as f:
        json.dump(sections, f, indent=2)

    # And save the filter intensities
    tab.write('adal-filter-predicted-rates.tab', format='ascii.tab', formats=col_fmts)


        

if __name__ == "__main__":
    argh.dispatch_command(main)
