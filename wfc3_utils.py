import os
import numpy as np

datapath = os.path.join(os.path.dirname(__file__), 
                        "WFC3-filters", "SystemThroughput")

WFC3_CONSTANT = 0.0840241 # counts cm^2 sr / erg / Ang / WFC3 pixel (used to be 0.29462)
GAIN = 1.5
ARCSEC_RADIAN = 180.0*3600.0/np.pi 

AIR_REFRACTIVE_INDEX = 1.000277 # @STP according to Wikipedia

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


def Ti(wav0, wavs, T):
    "Filter transmission at wavelength of line i (wav0)"
    return np.interp(wav0, wavs, T)


def Wtwid(wav0, wavs, T, kji=1.0):
    """Find W-twiddle for a given line of wavelength wav0
    with respect to a filter transmission curve T(wavs)"""
    # We are still missing the k_{j,i} term 
    return kji*Tm(T)*Wj(wavs, T)/Ti(wav0, wavs, T)


def ratio_coefficients(wav1=5755, wav2=6584, I="FQ575N", II="F658N", III="F547M"):
    wavs, T_I = get_filter(I, return_wavelength=True)
    T_II = get_filter(II)
    T_III = get_filter(III)

    T_1_III = Ti(wav1, wavs, T_III)
    T_2_III = Ti(wav2, wavs, T_III)
    T_1_I = Ti(wav1, wavs, T_I)
    T_2_II = Ti(wav2, wavs, T_II)

    Tm_I = Tm(T_I)
    Tm_II = Tm(T_II)
    Tm_III = Tm(T_III)
    
    W_I = Wj(wavs, T_I)
    W_II = Wj(wavs, T_II)
    W_III = Wj(wavs, T_III)

    return {
        "alpha_1": T_1_III/T_1_I,
        "alpha_2": T_2_III/T_2_II,
        "beta_1": Tm_I*W_I/(Tm_III*W_III),
        "beta_2": Tm_II*W_II/(Tm_III*W_III),
    }
    
