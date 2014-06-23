from astropy.table import Table
import argh
import json

linesets = {
    "diag-blue": ["3968",
                  "4359", "4363", "4368", "4388", "4438",
                  "4591",
                  "4631", "4634", 
                  "4639", "4642", "4649", "4651", "4658", "4662", "4667", "4676",
                  "4711", "4740"],
    "diag-green": ["4591",
                   "4639", "4642", "4649", "4651", "4662", "4676",
                   "4711", "4740",
                   "4959", "5007",
                   "5041", "5048", "5056",
                   "5518", "5538", "5555", "5577"],
    "diag-red": ["5876", "6678"]

}

def main(dbname="manu_spectral_fit_db.json"):
    with open(dbname) as f:
        db = json.load(f)
    tabs = {}
    for name, lines in linesets.items():
        colnames = ["x", "y"] + ["F"+line for line in lines]
        dtypes = [float]*len(colnames)
        tabs[name] = Table(names=colnames, dtypes=dtypes)

    for posname, posdata in db.items():
        for name, lines in linesets.items():
            wavband = name.split('-')[-1]
            if not posname.startswith(wavband):
                continue
            try:
                rowdata = [posdata.get(line, {"Flux": 0.0})["Flux"] for line in lines]
                tabs[name].add_row([posdata['x'], posdata['y']] + rowdata)
            except KeyError:
                pass

    for name, lines in linesets.items():
        tabs[name].write("manu-{}-fluxes.tab".format(name), format='ascii.tab')

if __name__ == "__main__":
    argh.dispatch_command(main)









