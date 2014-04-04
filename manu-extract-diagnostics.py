from astropy.table import Table
import argh
import json

linesets = {
    "diag-blue": ["4363",
                  "4591",
                  "4639", "4642", "4649", "4651", "4662", "4676",
                  "4711", "4740"],
    "diag-green": ["4591",
                   "4639", "4642", "4649", "4651", "4662", "4676",
                   "4711", "4740",
                   "4959", "5007",
                   "5518", "5538"]
}

def main(dbname="manu_spectral_fit_db.json"):
    with open(dbname) as f:
        db = json.load(f)
    tabs = {}
    for name, lines in linesets.items():
        colnames = ["x", "y"] + ["F"+line for line in lines]
        dtypes = [float]*len(colnames)
        tabs[name] = Table(names=colnames, dtypes=dtypes)

    for posdata in db.values():
        for name, lines in linesets.items():
            try:
                rowdata = [posdata[line]["Flux"] for line in lines]
                tabs[name].add_row([posdata['x'], posdata['y']] + rowdata)
            except KeyError:
                pass

    for name, lines in linesets.items():
        tabs[name].write("manu-{}-fluxes.tab".format(name), format='ascii.tab')

if __name__ == "__main__":
    argh.dispatch_command(main)
