import numpy as np
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as coord
import astropy.units as u
from astropy import wcs

def slit_profile(wav0=5754.6, width=14, trimblue=0, trimred=0, wblue=1.0, wred=1.0):
    """Extract emission line profiles from Adal spectra

    By default it takes a window of 14 pixels, centered on the line.
    Then another 7 pixels each side to find the local "continuum" on
    the blue and red sides, which is subtracted to get the integrated line emission. 

    Optional arguments allow width to be changed, or for pixels to be
    trimmed from the bg sections, or the weight of one or other to be
    changed.  This is necessary if there is another nearby emission line. 

    """
    i0 = np.argmin(np.abs(wavs - wav0))
    profile = hdu.data[:, i0-width//2:i0+width//2].mean(axis=1)
    bprofile = hdu.data[:, i0-width+trimblue:i0-width//2].mean(axis=1)
    rprofile = hdu.data[:, i0+width//2:i0+width-trimred].mean(axis=1)
    bgprofile = (wred*rprofile + wblue*bprofile)/(wblue+wred)
    return width*dwav*(profile - bgprofile), width*dwav*rprofile, width*dwav*bprofile

def set_coord(coordstr):
    return coord.ICRSCoordinates(coordstr=coordstr, unit=(u.hour, u.deg))

slitdata = {
    6: {
        "center": "05:35:15.2 -05:23:53.1",
        "PA": 72.0,
        "width": 1.03
    },
    5: {
        "center": "05:35:15.9 -05:23:50.3",
        "PA": 50.0,
        "width": 1.03
    },
}

filters = ["F658N", "F547M", "FQ575N"]
# Read in smoothed images
shdus = {fname: fits.open("full_{}_smooth12.fits".format(fname))[0]
         for fname in filters}

wny, wnx = shdus["F658N"].data.shape
w = wcs.WCS(shdus["F658N"].header)
# Find RA and DEC grid for the image
X, Y = np.meshgrid(np.arange(wnx), np.arange(wny))
RA, DEC = w.all_pix2world(X, Y, 0)

# change all NaNs to zeros
cropmask = np.ones((wny, wnx), dtype=bool)
for filter_ in filters:
    cropmask = cropmask & np.isfinite(shdus[filter_].data)
ims = {filter_: np.where(cropmask, shdus[filter_].data, 0.0) for filter_ in filters}



for islit in 5, 6:

    hdu = fits.open("Adal-Slits/zorip{:01d}rojo_1d.fits".format(islit))[1]

    nx, wav0, i0, dwav = [hdu.header[k] for k in ("NAXIS1", "CRVAL1",
                                                  "CRPIX1", "CD1_1")]
    wavs = wav0 + (np.arange(nx) - (i0 - 1))*dwav
    ny, dx = [hdu.header[k] for k in ("NAXIS2", "CD2_2")]
    x0, j0 = 0.0, 0.0
    xpos = x0 + (np.arange(ny) - (j0 - 1))*dx 
    xadal = 1.2*(xpos - 77.5) + 1.5

    profile6583, rprofile, bprofile = slit_profile(6583.4, width=18, wblue=0.0)
    profile6548, rprofile, bprofile = slit_profile(6547.9, wred=0.0)
    profile5755, rprofile, bprofile = slit_profile(5754.6)

    # Use range from 6050 to 6200 to find the continuum
    # This avoids any apparent lines
    # For the moment we ignore any color correction
    i1 = np.argmin(np.abs(wavs - 6050.0))
    i2 = np.argmin(np.abs(wavs - 6200.0))

    profile_cont = hdu.data[:,i1:i2].mean(axis=1)

    table = Table( data=[xadal], names=["Position"])
    table["F6583"] = profile6583
    table["F6548"] = profile6548
    table["F5755"] = profile5755
    table["Fcont"] = profile_cont

    table["W6583"] = profile6583/profile_cont
    table["W6548"] = profile6548/profile_cont
    table["W5755"] = profile5755/profile_cont

    slit_center = set_coord(slitdata[islit]["center"])
    slit_PA = slitdata[islit]["PA"]
    slit_width = slitdata[islit]["width"]

    RA0, DEC0 = slit_center.ra.deg, slit_center.dec.deg
    dRA = (RA - RA0)*3600*np.cos(np.radians(DEC0))
    dDEC = (DEC - DEC0)*3600

    COSPA, SINPA = np.cos(np.radians(slit_PA)), np.sin(np.radians(slit_PA))
    # Coordinates along and across slit
    xslit = dRA*SINPA + dDEC*COSPA
    yslit = -dRA*COSPA + dDEC*SINPA

    fits.PrimaryHDU(data=xslit,
                    header=hdu.header).writeto("Adal_xslit{}_north_pad.fits".format(islit),
                                               clobber=True)
    fits.PrimaryHDU(data=yslit,
                    header=hdu.header).writeto("Adal_yslit{}_north_pad.fits".format(islit),
                                               clobber=True)

    slitmask = np.abs(yslit) <= 0.5*slit_width
    print("Total pixels in slit:", np.sum(slitmask), "Solid angle:", np.sum(slitmask)*(0.04**2), "sq arcsec")
    R658, R658e = [], []
    R547, R547e = [], []
    R575, R575e = [], []
    xim = []
    for x in xadal:
        xmask = np.abs(-xslit - x) <= 0.6
        m = xmask & slitmask & cropmask
        n = m.sum()
        if n > 0:
            print("x =", x, "pixels = ", n, "area = ", n*(0.04**2), "sq arcsec")
            R658.append(ims["F658N"][m].mean())   
            R547.append(ims["F547M"][m].mean())
            R575.append(ims["FQ575N"][m].mean())
            R658e.append(ims["F658N"][m].std())   
            R547e.append(ims["F547M"][m].std())
            R575e.append(ims["FQ575N"][m].std())
            xim.append((-xslit*m).sum()/m.sum())
        else:
            R658.append(np.nan)
            R547.append(np.nan)
            R575.append(np.nan)
            R658e.append(np.nan)
            R547e.append(np.nan)
            R575e.append(np.nan)
            xim.append(np.nan)
        
    table["R658"] = R658
    table["R547"] = R547
    table["R575"] = R575
    table["R658e"] = R658e 
    table["R547e"] = R547e 
    table["R575e"] = R575e 
    table["xim"] = xim

    table.write("adal-slit{:01d}-EW.dat".format(islit), format="ascii.tab")
