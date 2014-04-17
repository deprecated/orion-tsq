
import numpy as np

def find_sweetspot_mask(fn, x, y):
    """Create a mask for Bob's so-called sweet spot"""
    sweetmask = np.ones_like(x).astype(bool)
    if 'Q' in fn.upper():
        # Common boundaries for all the quad filters
        theta = np.radians(56.0)
        s = -x*np.cos(theta) + y*np.sin(theta)
        sweetmask[s > 10.0] = False
        sweetmask[s < -70.0] = False
        theta = np.radians(-34.0)
        s = -x*np.cos(theta) + y*np.sin(theta)
        sweetmask[s > 82.0] = False
        sweetmask[s < 40.0] = False
    if fn.upper() in ['FQ672N', 'FQ674N']:
        # extra cut-off for [S II] filters
         sweetmask[x > -30.0] = False
    if fn.upper() in ['FQ575N', 'FQ436N']:
        # extra cut-off for [N II] and [O III] filters
        sweetmask[x < -75.0] = False

    return sweetmask
