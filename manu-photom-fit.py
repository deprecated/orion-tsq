from __future__ import print_function
import numpy as np
from astropy.table import Table
import lmfit
import json
from pathlib import Path
import argh
import sys 
sys.path.insert(0, '../../RingNebula/WFC3/2013-Geometry')
from photom_utils import model, \
    model_minus_data, \
    init_gauss_component, \
    LARGE_VALUE, init_poly_component
from manu_utils import sanitize_string


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
    if params['u0'+suffix].expr is not None:
        d.update(
            Wav_expr=params['u0'+suffix].expr,
            Sigma_expr=params['sigma'+suffix].expr,
        )
    

def store_poly_component(db, params, clabel="Power Law"):
    if not clabel in db:
        db[clabel] = {}
    db[clabel].update(
        C0=params["p0"].value, dC0=params["p0"].stderr,
        C1=params["p1"].value, dC1=params["p1"].stderr,
        C2=params["p2"].value, dC2=params["p2"].stderr,
    )


def store_all_components(db, params, gauss_components):
    for clabel in gauss_components:
        store_component(db, params, clabel)
    store_poly_component(db, params)
        

def fit_continuum(wavs, spec, cmask, npoly=4):
    cont_coeffs = np.polyfit(wavs[cmask], spec[cmask], npoly)
    return np.poly1d(cont_coeffs)(wavs)


def save_params_values(params, path):
    """Dump the parameter values to a file"""
    d = {k: params[k].value for k in params}
    with path.open("w") as f:
        json.dump(d, f, indent=2)
    

def tie_lines_together(params, lineA, lineB):
    """Set the parameters such that lineA is forced to have the same width
as lineB and the same radial velocity
    """
    suffixA = "_gauss_{}".format(lineA)
    suffixB = "_gauss_{}".format(lineB)
    wav_offset = params['u0'+suffixA].init_value - params['u0'+suffixB].init_value
    params['u0'+suffixA].expr = '{:s} + {:.3f}'.format('u0'+suffixB, wav_offset)
    params['sigma'+suffixA].expr = 'sigma'+suffixB


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

# Lines that are forced to have the same Delta Wav and Sigma as another line
tied_lines = {
    # O II lines
    "4651": "4639", "4662": "4639", "4649": "4639",
    # N III lines
    "4641": "4634",
    # N II lines
    "4643": "4631",
}

# Sky
drop_these = ["6398"]
# For Orion, we also drop He II and Mg I
drop_these += ["4542", "4563", "4571"]
# These N II and N III lines are right underneath the O III V1 multiplet
# Unfortunately, they make the line fitting be too degenerate
drop_these += ["4641", "4643"]

saturation_level = 25.0
# possibly_saturated = ["4959", "5007", "6563", "6583"]
possibly_saturated = []

positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"
wavrange_dir = Path("Manu-Data") / "WavRanges"

def main(pattern="*", rangelist="narrow", only=None, xtol=1.e-3, ftol=1.e-3, maxfev=0):
    """Fit Gaussians to all the lines in the spectra that match `pattern`"""
    wavranges = json.load(open('Manu-Data/wavrange-{}.json'.format(rangelist)))
    positions_paths = positions_dir.glob(pattern + ".json")
    for path in positions_paths:
        data = json.load(path.open())
        position_id = path.stem
        band = data["band"]
        wavs = np.array(data["wavs"])
        flux = np.array(data["mean"])
        cont = np.array(data["cont"])
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
            if only is not None:
                # If the --only option is set, check that wav is within this range
                if not wavmin <= float(only) <= wavmax:
                    # Otherwise, skip
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
                                         ubounds=(wav0-0.1, wav0+1.5),
                                         wbounds=(0.4, 1.2))
                )
                if clabel in tied_lines:
                    tie_lines_together(params, clabel, tied_lines[clabel])

            result = lmfit.minimize(model_minus_data, params, 
                                    args=(wavs[m], flux[m], gauss_components),
                                    xtol=xtol, ftol=ftol, maxfev=maxfev
            )
            print(result.message)

            # A full dump of the fit parameters each wav range (for later plotting)
            wavrange_subdir = wavrange_dir / position_id
            if not wavrange_subdir.is_dir():
                wavrange_subdir.mkdir(parents=True)
            savepath = wavrange_subdir / (sanitize_string(wav_id) + ".json")
            save_params_values(params, savepath)

            # And also save a dict for each emission line, with all relevant data
            store_all_components(fitdata, params, gauss_components)

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
                





if __name__ == "__main__":
    argh.dispatch_command(main)
