import json
from pathlib import Path
import argh
from astropy.table import Table

LIGHTSPEED = 2.99792458e5       # km/s

positions_dir = Path("Manu-Data") / "Positions"
fit_dir = Path("Manu-Data") / "LineFit"

tables_dir = Path("Manu-Data") / "Tables"

poscolumns = ['x', 'y']
linecolumns = ['Flux', 'dFlux', 'EW', 'dEW', 'V', 'dV', 'W', 'dW']

derived_velwidths = {'dV': 'dWav', 'W': 'Sigma', 'dW': 'dSigma'}

def main(wav: "Line wavelength, e.g., 4649", wavband: "Red, green, or blue"):
    positions_paths = positions_dir.glob(wavband + "*.json")
    linetable = Table(names=poscolumns + linecolumns,
                      dtype=[float]*(len(poscolumns) + len(linecolumns)))
    for path in positions_paths:
        position_id = path.stem
        linepath = fit_dir / position_id / "{}.json".format(wav)
        try:
            with linepath.open() as f:
                linedata = json.load(f)
        except FileNotFoundError:
            # skip positions that do not have this line
            continue
        with path.open() as f:
            posdata = json.load(f)
        linedata['V'] = (linedata['Wav'] - linedata['Wav0'])*LIGHTSPEED/linedata['Wav0']
        for vel, lam in derived_velwidths.items():
            linedata[vel] = linedata[lam]*LIGHTSPEED/linedata['Wav0']
        rowdata = [posdata[k] for k in poscolumns] + [linedata[k] for k in linecolumns]
        linetable.add_row(rowdata)

    if not tables_dir.is_dir():
        tables_dir.mkdir(parents=True)
    table_file = tables_dir / "{}-{}.tab".format(wav, wavband)
    linetable.write(str(table_file), format='ascii.tab')

if __name__ == "__main__":
    argh.dispatch_command(main)
