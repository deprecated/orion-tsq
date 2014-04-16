import json
from pathlib import Path
import argh
import wfc3_utils
import numpy as np
from astropy.table import Table

positions_dir = Path("Manu-Data") / "Positions"

fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N", "F645N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

col_names = ["Section", "x", "y"] + fns
col_dtypes = ["<U15", float, float] + [float]*len(fns)

bands = ["red", "green", "blue"]
wavcache = {}
Tcache = {"red": {}, "green": {}, "blue": {}}

def main():
    """Construct a table of per-filter fluxes from the Manu spectra. 

Uses the data written by manu-photom-select"""
    positions_paths = positions_dir.glob("*.json")
    tab = Table(names=col_names, dtype=col_dtypes)
    for path in positions_paths:
        with path.open() as f:
            data = json.load(f)
        print(path.stem)
        table_row = {"Section": path.stem, "x": data["x"], "y": data["y"]}
        for b in bands:
            if path.stem.startswith(b):
                band = b 
                exit
        if not band in wavcache:
            wavcache[band] = np.array(data["wavs"])
        spectrum = np.array(data["mean"]) + np.array(data["cont"])
        for fn in fns:
            if not fn in Tcache[band]:
                Tcache[band][fn] = wfc3_utils.get_interpolated_filter(fn,
                                                                      wavcache[band])
            table_row[fn] =  np.trapz(Tcache[band][fn]*spectrum, wavcache[band])
        tab.add_row(table_row)
    tab.write('manu-filter-fluxes.tab', format='ascii.tab')


        

if __name__ == "__main__":
    argh.dispatch_command(main)
