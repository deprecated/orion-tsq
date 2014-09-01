import os
import numpy as np

datapath = os.path.join(os.path.dirname(__file__), 
                        "WFC3-filters", "SystemThroughput")

WFC3_CONSTANT = 0.0840241 # counts cm^2 sr / erg / Ang / WFC3 pixel (used to be 0.29462)
GAIN = 1.5
ARCSEC_RADIAN = 180.0*3600.0/np.pi 

AIR_REFRACTIVE_INDEX = 1.000277 # @STP according to Wikipedia
LIGHTSPEED = 2.99792458e5       # c in km/s


# Adjustments to nominal values of the line transmission in a given band
line_transmission_adjustments = {
    # Defaults are weighted averages of ODH, Mesa-Delgado, & Ring
    (6731, "FQ674N"): 0.955,
    (6731, "F673N"): 0.974,
    (6716, "F673N"): 0.974,
    (6716, "FQ672N"): 0.930,
    (6583, "F658N"): 0.930,
    (6563, "F656N"): 1.013,
    (5755, "FQ575N"): 1.048,
    (5007, "F502N"): 0.953,
    (4861, "F487N"): 0.985,
    (4686, "F469N"): 1.350,
    (4363, "FQ437N"): 1.022,
    (4363, "FQ436N"): 0.940,
    (4340, "FQ436N"): 0.940,
}

# Accurate rest wavelengths
air_rest_wavelength = {
    4340: 4340.47,
    4363: 4363.21,
    4861: 4861.63,
    5007: 5006.84,
    5755: 5755.08,
    6563: 6562.79,
    6583: 6583.45,
    6716: 6716.44,
    6731: 6730.816
}

# Radial velocity of source wrt HST
# Default value is suitable for Orion Nebula with no HST-sun correction
# But note that earth-sun motion may be +/- 30 km/s
# And also HST-earth motion may be +/- 8 km/s
topocentric_velocity = 20.0     


def get_filter(fname, UVIS=1, return_wavelength=False):
    datafile = "{:s}.UVIS{:01d}.tab".format(fname.lower(), UVIS)
    fullpath = os.path.join(datapath, datafile) 
    data = np.genfromtxt(fullpath,
                      names=("row", "wavelength", "throughput"))
    if return_wavelength:
        return data["wavelength"], data["throughput"]
    else:
        return data["throughput"]


def get_interpolated_filter(fname, wavs, UVIS=1):
    """
    Rebin a WFC3 filter to the array of wavelengths `wavs`
    """
    filter_wavs, T = get_filter(fname, UVIS, return_wavelength=True)
    return np.interp(wavs, filter_wavs, T)

    
def Tm(T):
    "Maximum transmission of filter T"
    return T.max()


def Wj(wavs, T):
    "Rectangular width of filter T"
    return np.trapz(T, wavs)/Tm(T)


def vacuum_wavelength(wav0):
    "Convert from wav identifier to an accurate (0.01 Ang) vacuum wavelength"
    return air_rest_wavelength[wav0] \
        *AIR_REFRACTIVE_INDEX*(1.0 + topocentric_velocity/LIGHTSPEED)


def Ti(wav0, wavs, fname, T):
    "Filter transmission at wavelength of line i (wav0)"
    correction = line_transmission_adjustments.get((wav0, fname), 1.0)
    return correction*np.interp(vacuum_wavelength(wav0), wavs, T)


def Wtwid(wav0, wavs, fname, T, kji=1.0):
    """Find W-twiddle for a given line of wavelength wav0
    with respect to a filter transmission curve T(wavs)"""
    return kji*Tm(T)*Wj(wavs, T)/Ti(wav0, wavs, fname, T)


def ratio_coefficients(wav1=5755, wav2=6584, I="FQ575N", II="F658N", III="F547M"):
    wavs, T_I = get_filter(I, return_wavelength=True)
    T_II = get_filter(II)
    T_III = get_filter(III)

    T_1_III = Ti(wav1, wavs, III, T_III)
    T_2_III = Ti(wav2, wavs, III, T_III)
    T_1_I = Ti(wav1, wavs, I, T_I)
    T_2_II = Ti(wav2, wavs, II, T_II)
    T_1_II = Ti(wav1, wavs, II, T_II)
    T_2_I = Ti(wav2, wavs, I, T_I)

    Tm_I = Tm(T_I)
    Tm_II = Tm(T_II)
    Tm_III = Tm(T_III)
    
    W_I = Wj(wavs, T_I)
    W_II = Wj(wavs, T_II)
    W_III = Wj(wavs, T_III)

    return {
        "T2/T1": T_2_II/T_1_I,
        "alpha_1": T_1_III/T_1_I,
        "alpha_2": T_2_III/T_2_II,
        "beta_1": Tm_I*W_I/(Tm_III*W_III),
        "beta_2": Tm_II*W_II/(Tm_III*W_III),
        "gamma_1": T_1_II/T_1_I,
        "gamma_2": T_2_I/T_2_II,
    }
    


def find_line_ratio(filterset, R_I, R_II, R_III, k_I=1.0, k_II=1.0, naive=False):
    """Find the line ratio from a 3-filter set

If `naive` is True, then ignore the continuum and line contamination terms
"""
    wav1 = filterset["wav1"]
    wav2 = filterset["wav2"]
    wavs, T_I = get_filter(filterset['I'], return_wavelength=True)
    T_II = get_filter(filterset['II'])
    T_1_I = Ti(wav1, wavs, filterset['I'], T_I)
    T_2_II = Ti(wav2, wavs, filterset['II'], T_II)

    if naive:
        ratio = R_I/R_II
    else:
        contam_coeffs = ratio_coefficients(**filterset)
        alpha_1 = contam_coeffs["alpha_1"]
        alpha_2 = contam_coeffs["alpha_2"]
        beta_I = contam_coeffs["beta_1"]
        beta_II = contam_coeffs["beta_2"]
        gamma_1 = contam_coeffs["gamma_1"]
        gamma_2 = contam_coeffs["gamma_2"]
        
        ratio = (1.0 - alpha_2*beta_II*k_II)*R_I \
                + (alpha_2*beta_I*k_I - gamma_2)*R_II \
                + (gamma_2*beta_II*k_II - beta_I*k_I)*R_III
        ratio /= (alpha_1*beta_II*k_II - gamma_1)*R_I \
                 + (1.0 - alpha_1*beta_I*k_I)*R_II \
                 + (gamma_1*beta_I*k_I - beta_II*k_II)*R_III 

    ratio *= (wav2*T_2_II)/(wav1*T_1_I)
    return ratio

# Local Variables:
# wjh/elpy-virtual-environment: "~/anaconda/envs/py27"
# End:
