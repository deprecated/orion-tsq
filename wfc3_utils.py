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


def Wtwid(wav0, wavs, T):
    """Find W-twiddle for a given line of wavelength wav0
    with respect to a filter transmission curve T(wavs)"""
    Ti = np.interp(wav0, wavs, T)
    # We are still missing the k_{j,i} term 
    return Tm(T)*Wj(wavs, T)/Ti

