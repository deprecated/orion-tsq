from __future__ import print_function
import numpy as np
from astropy.io import fits
from astropy.table import Table
import lmfit
import json
from pathlib import Path
import sys 
sys.path.insert(0, '../../RingNebula/WFC3/2013-Geometry')
from photom_utils import model, \
    model_minus_data, \
    init_gauss_component, \
    LARGE_VALUE, init_poly_component


def store_component(db, params, clabel, ctype="gauss"):
    """
    Update db with a single spectrum component
    """
    if not clabel in db:
        db[clabel] = {}
    d = db[clabel]
    suffix = "_{}_{}".format(ctype, clabel)
    d.update(
             Species=species_dict.get(clabel, "Glow"),
             Flux=params['area'+suffix].value,
             dFlux=params['area'+suffix].stderr,
             Wav=params['u0'+suffix].value,
             dWav=params['u0'+suffix].stderr,
             Wav0=params['u0'+suffix].init_value,
             Sigma=params['sigma'+suffix].value,
             dSigma=params['sigma'+suffix].stderr,
             )
    if ctype == "gauss":
        d.update(
            Sat=params['saturation'+suffix].value,
            dSat=params['saturation'+suffix].stderr,
            )
    

def store_poly_component(db, params, clabel="Power Law"):
    if not clabel in db:
        db[clabel] = {}
    db[clabel].update(
        C0=params["p0"].value, dC0=params["p0"].stderr,
        C1=params["p1"].value, dC1=params["p1"].stderr,
        C2=params["p2"].value, dC2=params["p2"].stderr,
    )


def store_all_components(db, params):
    for clabel in gauss_components:
        store_component(db, params, clabel)
    store_poly_component(db, params)
        

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
brightlines = [6562.79, 4861.32, 4958.91, 5006.84]

species_dict = {}
for c in line_table:
    id_ = str(int(c['linewav']+0.5))
    species_dict[id_] = c['lineid']

wavranges = [
    ["Far Blue", 3960.0, 4200.0, "b"],
    ["FQ436N, FQ437N A", 4240.0, 4320.0, "b"],
    ["FQ436N, FQ437N B", 4320.0, 4400.0, "b"],
    ["FQ436N, FQ437N C", 4400.0, 4500.0, "b"],
    ["F469N A", 4560.0, 4680.0, "bg"],
    ["F469N B", 4680.0, 4800.0, "bg"],
    ["F487N", 4800.0, 4900.0, "g"],
    ["F502N", 4900.0, 5100.0, "g"], 
    ["F547M short", 5150.0, 5400.0, "g"],
    ["F547M mid", 5350.0, 5650.0, "g"],
    ["FQ575N, F547M long A", 5650.0, 5830.0, "r"],
    ["FQ575N, F547M long B", 5830.0, 6000.0, "r"],
    ["[O I] Red", 6000.0, 6330.0, "r"],
    ["F656N, F658N A", 6330.0, 6480.0, "r"],
    ["F656N, F658N B", 6530.0, 6650.0, "r"],
    ["F673N", 6640.0, 6760.0, "r"],
    ]

# Sky
drop_these = ["6398"]
# For Orion, we also drop He II and Mg I
drop_these += ["4542", "4563", "4571"]
saturation_level = 25.0
# possibly_saturated = ["4959", "5007", "6563", "6583"]
possibly_saturated = []

positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"
positions_paths = positions_dir.glob("*.json")
for path in positions_paths:
    data = json.load(path.open())
    position_id = path.stem
    band = data["band"]
    wavs = data["wavs"]
    flux = data["mean"]
    cont = data["cont"]
    sigma = data["std"]
    fitdata = {}
    print(position_id)
    # Find the F547M continuum if we can
    m = (wavs > 5100.0) & (wavs < 5850.0)
    cont_F547M = np.mean(wavs[m]*cont[m])
    data["F547M continuum lam Flam"] = cont_F547M

    # Now loop through the bands, fitting all the lines in each
    for wav_id, wavmin, wavmax, bands_covered in wavranges:
        if not band in bands_covered:
            # skip over wav ranges that are not in the current band
            continue
        print(wav_id, wavmin, wavmax)
        wav0s = np.array(line_table["linewav"])
        wav0s = wav0s[(wav0s > wavmin) & (wav0s < wavmax)]
        m = (wavs > wavmin) & (wavs < wavmax)

        params = lmfit.Parameters()
        gauss_components = []
        init_poly_component(params, [0.0, 0.0, 0.0])
        for wav0 in wav0s:
            clabel = str(int(wav0+0.5))
            if clabel in drop_these:
                continue
            if clabel in possibly_saturated:
                saturation = saturation_level
            else:
                saturation = LARGE_VALUE
            gauss_components.append(
                init_gauss_component(params, 1.0, wav0, 1.0, clabel, 
                                     ubounds=(wav0-3.0, wav0+3.0),
                                     wbounds=(0.5, 4.0), saturation=saturation)
            )

        result = lmfit.minimize(model_minus_data, params, 
                                args=(wavs[m], flux[m], gauss_components),
                                xtol=1e-4, ftol=1e-4,
        )
        print(result.message)
        
        store_all_components(fitdata, params)
        
        for i, c in enumerate(gauss_components):
            # Now calculate auxiliary data for each line
            linedata = fitdata[c]
            linedata["local continuum excess"] = model(float(c), params, [])
            # Select a small window around the line for measuring the continuum
            wav_min = linedata["Wav"] - linedata["Sigma"]
            wav_max = linedata["Wav"] + linedata["Sigma"]
            m = (wavs > wav_min) & (wavs < wav_max)
            # This is the mean continuum from the whole-spectrum fit
            cont_mean = cont[m].mean()
            linedata["global continuum"] = cont_mean
            # Calculate and store equivalent width
            # EWs are calculated without the local excess correction to
            # the continuum, since it just made things worse when I tried
            # it 
            linedata["EW"] = linedata["Flux"] / cont_mean
            linedata["dEW"] = linedata["dFlux"] / cont_mean
            # Calculate and store continuum color wrt F547M range
            linedata["Color"] = linedata["Wav"]*cont_mean / cont_F547M
            
            # And save it all to disk
            fit_db_dir = fit_dir / position_id 
            linepath = fit_db_dir / (c + ".json")
            if not fit_db_dir.is_dir():
                fit_db_dir.mkdir(parents=True)
            with linepath.open("w") as f:
                json.dump(linedata, f, indent=2, cls=NumpyAwareJSONEncoder)   
                


