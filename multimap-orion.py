from astropy.io import fits
import numpy as np

# surface brightness maps
S6584 = fits.open("north6584.fits")["SCI"]
S5755 = fits.open("north5755.fits")["SCI"]

hdr = dict(nii=S5755.header, oiii=S4363.header)

# weights (in seconds of exposure per pixel)
W4363 = fits.open("4363-drz-weight.fits")["SCI"]
W5755 = fits.open("5755-drz-weight.fits")["SCI"]

# list of rebinning factors
mlist = [1, 2, 4, 8, 16, 32, 64, 128]

# crop original to be multiple of largest bin width
ny, nx = S5007.data.shape
ny = mlist[-1]*(ny//mlist[-1])
nx = mlist[-1]*(nx//mlist[-1])
oiiiwin = slice(None, ny, None), slice(None, nx, None)
ny, nx = S6584.data.shape
ny = mlist[-1]*(ny//mlist[-1])
nx = mlist[-1]*(nx//mlist[-1])
niiwin = slice(None, ny, None), slice(None, nx, None)
win = dict(nii=niiwin, oiii=oiiiwin)


# Masking strategies  
# strategies = ["basic", "weight", "stdn", "erode"]
strategies = ["basic", "weight", "losig"]

ions = ["nii", "oiii"]


maskid = dict(nii="-nii", oiii="")
 
for strategy in strategies:
    # The images must be grabbed anew for each strategy, since the
    # downsampling ends up destroying them
    nebular = dict(nii = S6584.data[niiwin].copy(), oiii = S5007.data[oiiiwin].copy())
    auroral = dict(nii = S5755.data[niiwin].copy(), oiii = S4363.data[oiiiwin].copy())
    weight = dict(nii = W5755.data[niiwin].copy(), oiii = W4363.data[oiiiwin].copy())

    # Read in the mask array for each ion for this masking strategy
    goodmask = { ion: 
                 fits.open("mask{}-{}.fits".format(maskid[ion], strategy)
                             )[1].data[win[ion]]
                 for ion in ions }

    for m in mlist: 
        print
        print "Rebinning at {0} by {0}".format(m) 
        print
        suffix = "rebin{0:03d}x{0:03d}".format(m)

        for ion in "nii", "oiii":
            # First, save the arrays we have
            for label, image in [("nebular", nebular), 
                                 ("auroral", auroral),
                                 ("mask", goodmask),
                                 ("weight", weight)]:
                # But only after growing back to their original size
                zoomimage = oversample(image[ion]*goodmask[ion], m)
                hdu = fits.PrimaryHDU(zoomimage, hdr[ion])
                savefile = "{}-{}-{}-{}.fits".format(label, ion, strategy, suffix)
                print "Saving ", savefile
                hdu.writeto(savefile, clobber=True)
            # Second, rebin them to half the size
            [nebular[ion], auroral[ion]], goodmask[ion], weight[ion] = \
                    downsample([nebular[ion], auroral[ion]], goodmask[ion], 
                               weights=weight[ion], mingood=2, verbose=True)
