import numpy as np
import json
import sys
from astropy.table import Table
import wfc3_utils

def find_E_by_W(secdata, wavs_f, T_f, itarget=5755):
    result = Table(names=["Species", "line", "E/W", "efrac(E/W)"], dtype=['<U12', int, float, float])
    for lineid, data in secdata.items():
        if (isinstance(data, dict) 
            and "Wav0" in data 
            and lineid != str(itarget) 
            and not data["Species"].startswith("Sky")):
            result.add_row([data["Species"], int(lineid),
                            data["EW"]/wfc3_utils.Wtwid(data["Wav0"], wavs_f, T_f),
                            data["dEW"]/data["EW"]])
    result.sort("E/W")
    return result


fns = ["F469N", "F673N", "F487N", "F502N", 
       "FQ436N", "FQ437N", "F547M", "FQ575N", "F645N",
       "F656N", "F658N", "FQ672N", "FQ674N"]

wavtargets = {"F469N": "4658", "F673N": "6716", 
              "F487N": "4861", "F502N": "5007", 
              "FQ436N": "4340", "FQ437N": "4363", 
              "F547M": "5755", "FQ575N": "5755", "F645N": None,
              "F656N": "6563", "F658N": "6583", 
              "FQ672N": "6716", "FQ674N": "6731"}

wavf, F547M = wfc3_utils.get_filter("F547M", return_wavelength=True)
filts = {fn: wfc3_utils.get_filter(fn) for fn in fns}

db = json.load(open("odh_spectral_fit_db.json"))

factabs = {}
for fn in fns:
    print("Processing", fn, "...")
    iwav = wavtargets[fn]
    factabs[fn] = Table(names=["Section", "Sum(E/W)", 
                               "dSum", "Strongest", 
                               "E{}".format(iwav), "dE{}".format(iwav), "E/W {}".format(iwav),
                               "k{}".format(iwav), "kk{}".format(iwav), "F{}".format(iwav)
                           ], 
                        dtype=['<U7', float, float, '<U22'] + [float]*6)
    for section, secdata in db.items():
        factors = find_E_by_W(secdata, wavf, filts[fn], iwav)
        sum_e_w = factors["E/W"].sum()
        sum_errors = np.sqrt(np.sum((factors["E/W"]*factors["efrac(E/W)"])**2))
        if iwav is not None:
            target = secdata[iwav]
            # Quantities for target line
            E = target["EW"]
            dE = target["dEW"]
            E_W = E/wfc3_utils.Wtwid(target["Wav0"], wavf, filts[fn])
            k = target['Color']
            kk = k*(target["global continuum"]
                    + target["local continuum excess"])/target["global continuum"]
            F = target["Flux"]
        else:
            E, dE, E_W, k, kk, F = 0.0, 0.0, 0.0, 1.0, 1.0, 0.0
        factors["E/W"] *= 100/sum_e_w
        strongest = "{} {} {:.0f}%".format(*factors[-1].data)
        factabs[fn].add_row([section, sum_e_w, sum_errors, strongest, E, dE, E_W, k, kk, F])

    datafile = "odh-spectra-data-{}.tab".format(fn)
    factabs[fn].write(datafile, format='ascii.tab')
