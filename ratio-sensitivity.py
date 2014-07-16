import numpy as np
from wfc3_utils import find_line_ratio
import wfc3_utils
from astropy.io import fits
from astropy.convolution import convolve, Gaussian2DKernel
from matplotlib import pyplot as plt
import pyregion

fitsfilenames = {
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
    "F502N": "full_F502N_north_pad.fits",
    "FQ437N": "full_FQ437N_north_pad.fits"
}

# Mean and std of the color terms ktwiddle
color_terms_mean_sig = {
    "FQ674N": (1.00, 0.10), 
    "F673N":  (1.00, 0.10), 
    "FQ672N": (0.99, 0.10), 
    "F658N":  (1.04, 0.11), 
    "F656N":  (1.04, 0.11), 
    "FQ575N": (0.97, 0.03), 
    "F547M":  (1.03, 0.01), 
    "F502N":  (1.10, -0.04), 
    "F487N":  (1.13, -0.06), 
    "FQ437N": (1.41, -0.13), 
    "FQ436N": (1.87, -0.08), 
}

# Mean and std of the line transmission from the "new strange" calibration
transmission_mean_sig = {
    (6731, "FQ674N"): (0.908, 0.054), 
    (6731, "F673N"):  (0.982, 0.027), 
    (6716, "F673N"):  (0.982, 0.027), 
    (6716, "FQ672N"): (0.900, 0.070), 
    (6583, "F658N"):  (0.937, 0.113), 
    (6563, "F656N"):  (1.017, 0.057), 
    (5755, "FQ575N"): (1.058, 0.055), 
    (5007, "F502N"):  (0.970, 0.064), 
    (4861, "F487N"):  (0.983, 0.054), 
    (4686, "F469N"):  (1.350, 0.000), 
    (4363, "FQ437N"): (1.002, 0.109), 
    (4363, "FQ436N"): (0.942, 0.052), 
    (4340, "FQ436N"): (0.942, 0.052), 
}


def set_transmission(deviations={}):
    """Set up all transmission adjustments for use by find_line_ratio

Uses the values in `transmission_mean_sig`.  Optional argument
`deviations` is dict giving the number and sign of stdevs that we
deviate from the mean for each (line, filter) combination.  E.g.:
{(6716, "FQ672N"): 1.0, (6731, "FQ674N"): -1.0}.  Combos that are not
in `deviations` just get the mean value.

    """
    for combo, (mean, sigma) in transmission_mean_sig.items():
        dev = deviations.get(combo, 0.0)
        wfc3_utils.line_transmission_adjustments[combo] = mean + dev*sigma


def get_color_term(fname="FQ575N", kshift=0.0):
    """Return color term for filter `fname`, shifted `kshift` stdevs from mean"""
    mean, sigma = color_terms_mean_sig[fname]
    return mean + kshift*sigma


def get_fits_data(fn='FQ575N'):
    """Now also mask out unwanted regions with nans"""
    ions_for_filters_with_masks = {'FQ575N': 'nii', 'FQ672N': 'sii', 'FQ674N': 'sii'}
    hdu = fits.open(fitsfilenames[fn])[0]
    ion_maybe = ions_for_filters_with_masks.get(fn)
    if ion_maybe:
        include = pyregion.open("will-{}-sweet-spot.reg".format(ion_maybe))
        exclude = pyregion.open("will-{}-exclude.reg".format(ion_maybe))
        ssmask = include.get_mask(hdu=hdu) & (~exclude.get_mask(hdu=hdu))
        hdu.data[~ssmask] = np.nan
    return hdu.data

    
def deredden_nii_ratio(rnii, rhbha, balmer0=2.874):
    """Uses the Blagrave reddening law"""
    chb = -np.log10(balmer0*rhbha) / 0.220
    return rnii*10**(0.099*chb)


filtersets = {
    "[N II] 5755/6583": {"wav1": 5755, "wav2": 6583,
                         "I": "FQ575N", "II": "F658N", "III": "F547M"},
    "[S II] 6716/6731": {"wav1": 6716, "wav2": 6731,
                         "I": "FQ672N", "II": "FQ674N", "III": "F547M"}, 
    "Balmer 4861/6563": {"wav1": 4861, "wav2": 6563,
                         "I": "F487N", "II": "F656N", "III": "F547M"},
}

# First do the color terms - we assume that they all vary in lockstep
# due to variation in the continuum slope.  This is why we have given
# the sig a negative value for filters that are on the blue side
colorsets = dict([["normal", 0], ["blue", -1], ["red", +1]])

# Dict to store the ratio images 
ratios = {}

# Use all mean values for T_ij
set_transmission()
for ratio_name, filterset in filtersets.items():
    FI, FII, FIII = [filterset[J] for J in ("I", "II", "III")]
    RI = get_fits_data(FI)
    RII = get_fits_data(FII)
    RIII = get_fits_data(FIII)
    for kname, kshift in colorsets.items():
        kI = get_color_term(FI, kshift)
        kII = get_color_term(FII, kshift)
        kIII = get_color_term(FIII, kshift)
        ratios[(ratio_name, kname)] = find_line_ratio(filterset, RI, RII, RIII,
                                                      k_I=kI/kIII, k_II=kII/kIII,
                                                      naive=False)

# Plot the histograms for the sensitivity to ktwiddle
pltfile = "ratio-sensitivity-ktwiddle.pdf"
plt_title = "Sensitivity to uncertainty in color terms"
xname, yname, zname = "[S II] 6716/6731", "[N II] 5755/6583", "Balmer 4861/6563"
xmin, xmax, ymin, ymax = 0.4, 0.8, 0.0, 0.05
snii = fits.open("full_F658N_north_pad.fits")[0].data
fig = plt.figure(figsize=(7,7))
GAMMA = 1.5
for kname in colorsets.keys():
    x, y, z = [ratios[(rname, kname)] for rname in (xname, yname, zname)] 
    y = deredden_nii_ratio(y, z)
    m = np.isfinite(x) & np.isfinite(y)
    H, xedges, yedges = np.histogram2d(x[m], y[m], 100,
                                       [[xmin, xmax], [ymin, ymax]],
                                       weights=snii[m])
    H = convolve(H, Gaussian2DKernel(1.0))
    if kname == "normal":
        plt.imshow((H.T)**(1.0/GAMMA), extent=[xmin, xmax, ymin, ymax],
                   interpolation='none', aspect='auto', origin='lower', 
                   cmap=plt.cm.gray_r, alpha=1.0)
    else:
        plt.contour(H.T, extent=[xmin, xmax, ymin, ymax],
                    colors=kname, origin='lower')

plt.xlabel(xname)
plt.ylabel(yname)
plt.grid()
plt.title(plt_title)
plt.axis([xmin, xmax, ymin, ymax])
fig.savefig(pltfile)


# Now do the sensitivity to systematic calibration errors
combos = [
    (6731, "FQ674N"), 
    (6716, "FQ672N"), 
    (6583, "F658N"), 
    (6563, "F656N"), 
    (5755, "FQ575N"), 
    (4861, "F487N"), 
]
calibsets = {"mean": 0, "green": -1, "magenta": +1}
for combo in combos:
    pltfile = "ratio-sensitivity-T{}-{}.pdf".format(*combo)
    plt_title = "Sensitivity to uncertainty in T({}, {})".format(*combo)
    fig = plt.figure(figsize=(7,7))
    for ratio_name, filterset in filtersets.items():
        FI, FII, FIII = [filterset[J] for J in ("I", "II", "III")]
        RI = get_fits_data(FI)
        RII = get_fits_data(FII)
        RIII = get_fits_data(FIII)
        kI = get_color_term(FI, 0.0)
        kII = get_color_term(FII, 0.0)
        kIII = get_color_term(FIII, 0.0)
        for cname, cshift in calibsets.items():
            set_transmission(deviations={combo: cshift})
            ratios[(ratio_name, cname)] = find_line_ratio(filterset,
                                                          RI, RII, RIII,
                                                          k_I=kI/kIII,
                                                          k_II=kII/kIII,
                                                          naive=False)
    for cname in calibsets.keys():
        x, y, z = [ratios[(rname, cname)] for rname in (xname, yname, zname)] 
        y = deredden_nii_ratio(y, z)
        m = np.isfinite(x) & np.isfinite(y)
        H, xedges, yedges = np.histogram2d(x[m], y[m], 100,
                                           [[xmin, xmax], [ymin, ymax]],
                                           weights=snii[m])
        H = convolve(H, Gaussian2DKernel(1.0))
        if cname == "mean":
            plt.imshow((H.T)**(1.0/GAMMA), extent=[xmin, xmax, ymin, ymax],
                       interpolation='none', aspect='auto', origin='lower', 
                       cmap=plt.cm.gray_r, alpha=1.0)
        else:
            plt.contour(H.T, extent=[xmin, xmax, ymin, ymax],
                        colors=cname, origin='lower')

    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.title(plt_title)
    plt.grid()
    plt.axis([xmin, xmax, ymin, ymax])
    fig.savefig(pltfile)
