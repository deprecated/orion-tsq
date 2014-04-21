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

col_names = ["Section", "x", "y", "pointing", "aperture"] + fns
col_dtypes = ["<U15", float, float, int, int] + [float]*len(fns)

bands = ["red", "green", "blue"]
wavcache = {}
Tcache = {"red": {}, "green": {}, "blue": {}}

WFC3_CONSTANT = 0.29462         # counts cm^2 sr / erg / Ang / pixel
FIBER_AREA_SQARCSEC = np.pi * (2.69/2.0)**2  # square arcseconds
ARCSEC_RADIAN = 180.0*3600.0/np.pi 
FIBER_AREA_SR = FIBER_AREA_SQARCSEC / ARCSEC_RADIAN**2
# The line fluxes were in units of erg/s/cm2/AA/fiber, 
# but have already been multiplied by 1e15 
FACTOR = 1.e-15*WFC3_CONSTANT/FIBER_AREA_SR

def main(choice='mean'):
    """Construct a table of per-filter fluxes from the Manu spectra. 

Uses the data written by manu-photom-select"""
    positions_paths = positions_dir.glob("*.json")
    tab = Table(names=col_names, dtype=col_dtypes)
    for path in positions_paths:
        with path.open() as f:
            data = json.load(f)
        print(path.stem)
        table_row = {"Section": path.stem, "x": data["x"], "y": data["y"],
                     "pointing": data["pointing"], "aperture": data["aperture"]}
        for b in bands:
            if path.stem.startswith(b):
                band = b 
                exit
        if not band in wavcache:
            wavcache[band] = np.array(data["wavs"])
        spectrum = np.array(data[choice])
        if choice == 'mean':
            # The mean is the only one that has had the continuum removed
            spectrum += np.array(data["cont"])
        for fn in fns:
            if not fn in Tcache[band]:
                Tcache[band][fn] = wfc3_utils.get_interpolated_filter(fn,
                                                                      wavcache[band])
            # Integrate lambda I_lambda T_lambda 
            integrand = Tcache[band][fn]*spectrum*wavcache[band]
            table_row[fn] = FACTOR*np.trapz(integrand, wavcache[band])
        tab.add_row(table_row)
    tab.write('manu-filter-predicted-rates.tab', format='ascii.tab')


        

if __name__ == "__main__":
    argh.dispatch_command(main)
