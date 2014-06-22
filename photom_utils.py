from __future__ import print_function
import numpy as np
import scipy.stats

verbose = False

def profile(distro, u, area=1.0, u0=0.0, sigma=1.0, du=None):
    """Generic profile with total area under the profile of area,
    centered on u0 and with RMS width sigma

    First argument `distro` should be scipy.stats distribution 
    with .pdf and .cdf methods defined. 

    Note that u, area, u0, sigma can all be 2-d arrays

    If optional argument du is present it is the velocity cell size
    and the profile is returned averaged over the cell

    """
    if du is None:
        # straightforward evaluation of the function
        return area*distro.pdf(u, loc=u0, scale=sigma)
    else:
        # average over the velocity cell
        return area*(distro.cdf(u+0.5*du, loc=u0, scale=sigma) -
                     distro.cdf(u-0.5*du, loc=u0, scale=sigma))/du
 
   
LARGE_VALUE = 1.e30
def gauss(u, area=1.0, u0=0.0, sigma=1.0, du=None, saturation=LARGE_VALUE):
    value = profile(scipy.stats.norm, u, area, u0, sigma, du)
    if saturation is not None:
        value[value > saturation] = saturation
    return value


def lorentz(u, area=1.0, u0=0.0, sigma=1.0, du=None):
    return profile(scipy.stats.cauchy, u, area, u0, sigma, du)

def init_gauss_component(params, area, u0, sigma, label, ubounds=(None, None), wbounds=(None, None), saturation=None):
    if verbose: 
        print("Initializing Gaussian component ", label)
    umin, umax = ubounds
    wmin, wmax = wbounds
    params.add("area_gauss_" + label, value=area, min=0.0)
    params.add("u0_gauss_" + label, value=u0, min=umin, max=umax)
    params.add("sigma_gauss_" + label, value=sigma, min=wmin, max=wmax)
    if saturation is not None and saturation != LARGE_VALUE:
        params.add("saturation_gauss_" + label, value=saturation, min=0.1*saturation, max=10*saturation)
    else:
        params.add("saturation_gauss_" + label, value=saturation, vary=False)
    return label


def init_lorentz_component(params, area, u0, sigma, label, ubounds=(None, None), wbounds=(None, None)):
    if verbose: 
        print("Initializing Lorentzian component ", label)
    umin, umax = ubounds
    wmin, wmax = wbounds
    params.add("area_lorentz_" + label, value=area, min=0.0)
    params.add("u0_lorentz_" + label, value=u0, min=umin, max=umax)
    params.add("sigma_lorentz_" + label, value=sigma, min=wmin, max=wmax)
    return label


MAX_NPOLY = 4

def init_poly_component(params, pcoeffs):
    assert(len(pcoeffs) <= MAX_NPOLY)
    if verbose:
        print("Initializing polynomial component")
    for i, p in enumerate(pcoeffs):
        params.add("p{}".format(i), value=p)
    return "Poly"


def model(wav, params, gauss_components, 
          lorentz_components=[], dwav=2.0, initial=False):
    """Summation of various components"""
    total = np.zeros_like(wav)
    for ABC in gauss_components:
        if initial: 
            area = params["area_gauss_" + ABC].init_value
            wav0 = params["u0_gauss_" + ABC].init_value
            sigma = params["sigma_gauss_" + ABC].init_value
            saturation = params["saturation_gauss_" + ABC].init_value
        else:
            area = params["area_gauss_" + ABC].value
            wav0 = params["u0_gauss_" + ABC].value
            sigma = params["sigma_gauss_" + ABC].value
            saturation = params["saturation_gauss_" + ABC].value
        total += gauss(wav, area, wav0, sigma, dwav, saturation)
    for ABC in lorentz_components:
        if initial: 
            area = params["area_lorentz_" + ABC].init_value
            wav0 = params["u0_lorentz_" + ABC].init_value
            sigma = params["sigma_lorentz_" + ABC].init_value
        else:
            area = params["area_lorentz_" + ABC].value
            wav0 = params["u0_lorentz_" + ABC].value
            sigma = params["sigma_lorentz_" + ABC].value
        total += lorentz(wav, area, wav0, sigma, dwav)

    # Add a polynomial if present
    pcoeffs = [0.0]*MAX_NPOLY
    for i in range(MAX_NPOLY):
        id_ = "p{}".format(i)
        if id_ in params:
            if initial:
                pcoeffs[i] = params[id_].init_value
            else:
                pcoeffs[i] = params[id_].value
    p = np.poly1d(pcoeffs[::-1]) # highest order goes first in poly1d constructor
    total += p(wav)

    return total


def model_minus_data(params, wavs, data, 
                     gauss_components, lorentz_components=[], du=2.0):
    """Function to minimize"""
    return model(wavs, params, gauss_components, lorentz_components, du) - data


def model_minus_data_over_sigma(params, wavs, data, sigma, 
                                gauss_components, lorentz_components=[], 
                                du=2.0):
    """Another function to minimize"""
    return (model(wavs, params, gauss_components, lorentz_components, du) - data)/sigma


