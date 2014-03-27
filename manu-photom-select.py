from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.table import Table
import json
import sys 
from pathlib import Path


def fit_continuum(wavs, spec, cmask, npoly=4):
    cont_coeffs = np.polyfit(wavs[cmask], spec[cmask], npoly)
    return np.poly1d(cont_coeffs)(wavs)

class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray) and obj.ndim <= 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# Read in the emission line rest wavelengths
line_table = Table.read("line-wavelengths-orion.tab", 
    format="ascii.no_header", delimiter="\t",
    names=('lineid', 'linewav')
    )

# Read in list of line-free wav ranges for the continuum fitting
cont_table = Table.read("clean-continuum-ranges.tab", 
    format="ascii.no_header", delimiter="\t",
    names=('wav1', 'wav2')
    )

# Box that comfortably covers the sweet spot
box_x, box_y = -43.0, -48.0     # Center of box in arcsec: dRA, dDEC
box_w, box_h = 60.0, 60.0       # Box width and height, in arcsec
# box_w, box_h = 3.0, 3.0       # Box width and height, in arcsec
THRESH = 1.e-13                 # Threshold for possible saturation of long exposures
brightlines = [6562.79, 4861.32, 4340.47, 4958.91, 5006.84]

sections = {}
wavs = {}
cmask = {}
# This is different from the slit observations, since each band is done separately
for longband, band in ("red", "r"), ("green", "g"), ("blue", "b"):
    thdu = {
        "long": fits.open("Manu-Data/M42_{}/table_M42_l{}.fits".format(longband, band))[1],
        "short": fits.open("Manu-Data/M42_{}/table_M42_s{}.fits".format(longband, band))[1],
    }
    hdu = {
        "long": fits.open("Manu-Data/M42_{}/M42_l{}.fits".format(longband, band))[0],
        "short": fits.open("Manu-Data/M42_{}/M42_s{}.fits".format(longband, band))[0],
    }

    # First filter out fibers based on position.  We will take a 60 x
    # 60 arcsec box, centered on 5:35:13.592, -5:24:11.04
    mask = (np.abs(thdu["long"].data.dRA - box_x) <= box_w/2) &  \
           (np.abs(thdu["long"].data.dDEC - box_y) <= box_h/2)
    # Collect only the masked fibers
    tabdata = thdu["long"].data[mask]
    tabdata_s = thdu["short"].data[mask]
    specdata = hdu["long"].data[mask]
    specdata_s = hdu["short"].data[mask]

    # Set up wavelength coordinates - Angstrom
    nx, wav0, i0, dwav = [hdu["long"].header[k] for k in 
                          ("NAXIS1", "CRVAL1", "CRPIX1", "CDELT1")]
    wavs[band] = wav0 + (np.arange(nx) - (i0 - 1))*dwav 

    # Use short exposure for the brightest lines
    brightmask = np.zeros_like(wavs[band]).astype(bool)
    for wav0 in brightlines:
        brightmask[np.abs(wavs[band] - wav0) <= 5.0] = True

    # Create a wavelength mask containing clean continuum regions
    cmask[band] = np.zeros_like(wavs[band], dtype=bool)
    for wav1, wav2 in cont_table:
        cmask[band] = cmask[band] | ((wavs[band] > wav1) & (wavs[band] < wav2))

    # Create one section for each fiber position in each of the three bands
    for spectrum, spectrum_s, metadata in zip(specdata, specdata_s, tabdata):
        # Key is formed from the band and the position, in 1/10 of
        # arcsec: e.g., green-0013-0104 for (dRA, dDEC) = (-1.3,
        # -10.4)
        key = "{:s}{:+05d}{:+05d}".format(longband,
                                          int(10*metadata["dRA"]),
                                          int(10*metadata["dDEC"]))
        section = {}
        sections[key] = section
        section["x"] = float(metadata["dRA"])
        section["y"] = float(metadata["dDEC"])
        section["aperture"] = int(metadata["id_ap"])
        section["band"] = band
        # Multiply by 1e15 as with the Helix to make the spectra of order unity for weak lines
        section["mean"] = 1.e15*np.where(brightmask, spectrum_s, spectrum)/metadata["factor2"]
        # We don't have a good estimate of the std of the data - so make something up!
        section["std"] = 0.01*np.ones_like(spectrum)
        # Fit continuum to the clean wav ranges
        section["cont"] = fit_continuum(wavs[band], section["mean"], cmask[band], npoly=2)
        section["mean"] -= section["cont"]
        section["wavs"] = wavs[band]


dbdir = Path("Manu-Data") / "Sections"
if not dbdir.is_dir():
    dbdir.mkdir(parents=True)

for secname, section in sections.items():
    p = dbdir / (secname + ".json")
    with p.open("w") as f:
        json.dump(section, f, indent=2, cls=NumpyAwareJSONEncoder)

