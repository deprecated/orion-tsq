from __future__ import print_function
import numpy as np
import json
from pathlib import Path
import lmfit
import argh
from astropy.table import Table
from matplotlib import pyplot as plt
import sys
sys.path.insert(0, '../../RingNebula/WFC3/2013-Geometry')
from photom_utils import model
from manu_utils import sanitize_string


def load_params_values(path):
    """Load the lmfit parameter values from a JSON file"""
    with path.open() as f:
        d = json.load(f)
    params = lmfit.Parameters()
    for k, v in d.items():
        params.add(k, value=v)
    return params


# Read in the emission line rest wavelengths
line_table = Table.read("line-wavelengths-orion.tab", 
    format="ascii.no_header", delimiter="\t",
    names=('lineid', 'linewav')
    )
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

positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"
wavrange_dir = Path("Manu-Data") / "WavRanges"
plot_dir = Path("Manu-Data") / "Plots"

def main(pattern="*", line_pattern="*", rangelist="narrow", remake=False):
    """Plot graphs of fits to PPAK spectra"""
    positions_paths = positions_dir.glob(pattern + ".json")
    with open('Manu-Data/wavrange-{}.json'.format(rangelist)) as f:
        wavranges = json.load(f)
    for path in positions_paths:
        with path.open() as f:
            data = json.load(f)
        position_id = path.stem
        print(position_id)
        fit_subdir = fit_dir / position_id
        plot_subdir = plot_dir / position_id
        if not plot_subdir.is_dir():
            plot_subdir.mkdir(parents=True)

        band = data["band"]
        wavs = np.array(data["wavs"])
        flux = np.array(data["mean"])
        cont = np.array(data["cont"])
        
        for wav_id, wavmin, wavmax, bands_covered in wavranges:
            if not band in bands_covered:
                continue
            m = (wavs > wavmin) & (wavs < wavmax)
            wavrange_subdir = wavrange_dir / position_id
            loadpath = wavrange_subdir / (sanitize_string(wav_id) + ".json")
            params = load_params_values(loadpath)
            gauss_components = [p.split('_')[-1] for p in params 
                                if p.startswith('area_gauss')]

            plotpath = plot_subdir / (sanitize_string(wav_id) + ".pdf")
            if plotpath.exists() and not remake:
                # Only replace existing plots if the --remake flag is specified
                continue
            fig, ax = plt.subplots(1, 1)
            ax.plot(wavs[m], cont[m]+ flux[m], label="data", lw=1.5, alpha=0.6)
            ax.plot(wavs[m], cont[m] + model(wavs[m], params, gauss_components),
                     "r", label="fit", lw=1.5, alpha=0.6)
            # plot the global continuum fit
            ax.plot(wavs[m], cont[m], ":k", label="global cont", lw=2, alpha=0.3)
            # and the continuum with local excess included
            ax.plot(wavs[m], cont[m] + model(wavs[m], params, []),
                    "--k", label="local cont", lw=2, alpha=0.3)
            for i, c in enumerate(sorted(gauss_components)):
                fit_path = fit_subdir / (c + ".json")
                with fit_path.open() as f:
                    cdata = json.load(f)
                ax.annotate("{} {}".format(species_dict[c], c), 
                             (cdata["Wav0"], cdata["global continuum"]), 
                             xytext=(0, -14*(1 + (i % 3))), 
                             textcoords="offset points", 
                             ha="center", va="top", fontsize="x-small", 
                             arrowprops={"arrowstyle": "->", "facecolor": "red"})
            ax.minorticks_on()
            ax.grid(ls='-', c='b', lw=0.3, alpha=0.3)
            ax.grid(ls='-', c='b', lw=0.3, alpha=0.05, which='minor')
            ymax = 1.5*np.max(flux[m] + cont[m])
            ymin = min(0.0, np.min(cont[m]) - 0.5*np.max(flux[m]))
            ax.set_ylim(ymin, ymax)
            # ax.set_ylim(ymin/1.1, 1.1*ymax)
            # ax.set_yscale('log')
            legtitle = "{} :: Fiber = {}".format(wav_id, position_id)

            legend = ax.legend(title=legtitle,
                               fontsize="small", ncol=4, loc="upper left")
            legend.get_title().set_fontsize("small")     
            
            ax.set_xlabel("Wavelength")
            ax.set_ylabel("Flux")

            fig.subplots_adjust(left=0.15, right=0.96, bottom=0.15, top=0.98)
            fig.set_size_inches(5, 4)

            fig.savefig(str(plotpath))
            plt.close(fig)


if __name__ == "__main__":
    argh.dispatch_command(main)
