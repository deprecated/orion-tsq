import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.table
from astropy.table import Table
import argh
from pathlib import Path
from argh.decorators import arg
sys.path.append("/Users/will/Work/RubinWFC3/Tsquared/")
import wfc3_utils

wavtargets = {"F469N": "4713",
              "F673N": "6716", 
              "F487N": "4861",
              "F502N": "5007", 
              "FQ436N": "4340",
              "FQ437N": "4363", 
              "F547M": "5755",
              "FQ575N": "5755",
              "F645N": None,
              "F656N": "6563",
              "F658N": "6583", 
              "FQ672N": "6716",
              "FQ674N": "6731"}

wavf, F547M = wfc3_utils.get_filter("F547M", return_wavelength=True)
filts = {fn: wfc3_utils.get_filter(fn) for fn in wavtargets.keys()}


def get_exact_wav(wavid):
    key = int(wavid)
    wavtab = Table.read('line-wavelengths.tab', 
                        format='ascii.no_header',
                        delimiter='\t',
                        names=['Ion', 'Wav']
    )
    m =  (wavtab['Wav'] + 0.5).astype(int) == key
    assert m.sum() == 1, "Need exactly one wav match, but got these: " + str(wavtab[m])
    wav_exact, = wavtab['Wav'][m]
    return wav_exact

    
def rqq(wav_line, wavs, TN, TW):
    from wfc3_utils import Tm, Wj, Wtwid
    r0 = Tm(TN)*Wj(wavs, TN)/(Tm(TW)*Wj(wavs, TW))
    q1 = (1./Wtwid(wav_line, wavs, TN) - 1./Wtwid(wav_line, wavs, TW))
    q2 = 1./(Wtwid(wav_line, wavs, TW) - Wtwid(wav_line, wavs, TN))
    return r0, q1, q2


def prelaunch_ratio(EW, r0, q1, q2):
    return r0*(1.0 + q1*EW*(1.0 - q2*EW))


def get_spectab(fname="FQ575N", dataset="odh"):
    """Read in the table of spectra for a given filter"""
    return Table.read("{}-spectra-data-{}.tab".format(dataset, fname), 
                      format="ascii.tab", 
                      fill_values=('--', 0.0))


def get_filtertab(f1, f2, dataset="odh"):
    """Read in table of filter brightnesses"""
    tabfile = dataset + "_calibration_db.tab"
    ftab = Table.read(tabfile, format="ascii.tab", fill_values=('--', 0.0))
    assert f1 in ftab.colnames, f1 + " not in " + tabfile
    assert f2 in ftab.colnames, f2 + " not in " + tabfile
    try:
        selection = ftab['PA', 'Section', 'x0', f1, f2]
    except ValueError:
        selection = ftab['x', 'y', 'Section', f1, f2]
    return selection


def find_sweetspot_mask(x, y):
    """Create a mask for Bob's so-called sweet spot"""
    sweetmask = np.ones_like(x).astype(bool)
    theta = np.radians(56.0)
    s = -x*np.cos(theta) + y*np.sin(theta)
    sweetmask[s > 10.0] = False
    sweetmask[s < -70.0] = False
    theta = np.radians(-34.0)
    s = -x*np.cos(theta) + y*np.sin(theta)
    sweetmask[s > 82.0] = False
    sweetmask[s < 40.0] = False
    return sweetmask
    

def plot_ew_ratio(tabs, wav, f1, f2,
                  fixedcolor=None,
                  colorstrategy="average", simple=False, logscale=True,
                  ymax=None, alpha=0.4):
    """Make a plot of Filter Ratio versus equivalent width"""
    snscale = {"odh": 1.0, "manu": 0.03, "adal": 1.0, "ring": 1.1}
    plotcolor = {"odh": "c", "manu": "r", "adal": "g", "ring": "y"}
    zorder = {"odh": 100, "manu": 0, "adal": 10, "ring": 50}
    r0, q1, q2 = rqq(get_exact_wav(wav), wavf, filts[f1], filts[f2])
    fig, ax = plt.subplots(1, 1)
    for dataset, tab in tabs.items(): 
        xo = tab['E'+wav]
        xe = 3*tab['dE'+wav]

        if fixedcolor is not None:
            kcolor = fixedcolor
            kerr = 0.0
        elif colorstrategy == "global":
            kcolor = tab['k'+wav]
            kerr = 0.0
        elif colorstrategy == "local":
            kcolor = tab['kk'+wav]
            kerr = 0.0
        else:
            kcolor = 0.5*(tab['k'+wav] + tab['kk'+wav])
            kerr = 0.5*np.abs((tab['k'+wav] - tab['kk'+wav])/tab['k'+wav])
        missing_values = getattr(tab['Sum(E/W)_2'], 'mask', np.array([False]))
        if  np.any(~missing_values):
            EW2 = tab['Sum(E/W)_2']
            dEW2 = tab['dSum_2']
        else:
            # Case where there is no filter2 spectra data for any position
            EW2 = 0.01
            dEW2 = 0.001
        yo = tab[f1]*(1.0 + EW2)/(tab[f2]*kcolor)

        if not simple:
            # Apply the correction for contaminating lines in the narrow filter
            yo -= r0*tab['Sum(E/W)_1']

        ye = tab[f1]*dEW2/(tab[f2]*kcolor)
        ye = np.sqrt(ye**2 + (kerr*yo)**2)
        snr = 2*np.sqrt((xo/xe)*(yo/ye))
        #snr = bigtab['W'+wav]/bigtab['dW'+wav]
        if 'x' in tab.colnames and 'y' in tab.colnames:
            d = np.hypot(tab['x'], tab['y'])
        else:
            y0 = np.array([float(s[1:3]) for s in tab['Section']])
            d = np.hypot(tab['x0'], y0)

        mask = np.ones_like(xo).astype(bool)
        # for column in f1, f2, 'Sum(E/W)_1', 'Sum(E/W)_2':
        for column in f1, f2, 'Sum(E/W)_1':
            try:
                mask = ~tab[column].mask & mask
            except AttributeError:
                pass

        if 'x' in tab.colnames:
            sm = find_sweetspot_mask(tab['x'], tab['y']) 
        else:
            sm = np.ones_like(tab[f1]).astype(bool)
        sm = sm & mask
        # ax.errorbar(xo[sm], yo[sm], xerr=xe[sm], yerr=ye[sm], fmt=None, zorder=0, alpha=alpha)
        ax.scatter(xo[sm], yo[sm], c=plotcolor[dataset],
                          s=snscale[dataset]*snr[sm], alpha=alpha,
                          zorder=zorder[dataset]
        )
        
        if dataset == "ring":
            xmax = xo[sm].max()*1.1
            print("Maximum", xo[sm].max(), yo[sm][xo[sm].argmax()])
        if dataset == "manu":
            xmin = xo[sm].min()/1.1

    x = np.linspace(xmin, xmax, 300)
    ax.plot(x, prelaunch_ratio(x, r0, q1, q2), label="Pre-launch calibration")
    ax.plot(x, prelaunch_ratio(x, 1.2*r0, q1, q2), label="1.2 x r0")
    ax.plot(x, prelaunch_ratio(x, r0, 1.2*q1, q2), label="1.2 x q1")
    ax.plot(x, prelaunch_ratio(x, 1.1*r0, 1.1*q1, q2), label="1.1 x r0, 1.1 x q1")
    # p = np.poly1d(np.polyfit(xo[mask], yo[mask], 1))
    # y0 = p(x)
    # ax.plot(x, y0, "r-", label="Linear fit")

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(xmin, xmax)

    # cbar = fig.colorbar(scat)
    # cbar.set_label("Distance from star, arcsec")
    # #cbar.set_label("F547M brightness")
    # cbar.solids.set_edgecolor("face")

    ax.set_xlabel(r'Equivalent width: $E_{{{}}}$, Angstrom'.format(wav), fontsize='x-large')
    
    ylabel = fmt_ylabel(f1, f2, simple)
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.minorticks_on()
    ax.legend(loc="upper left")
    ax.grid(ls='-', c='b', lw=0.3, alpha=0.3)
    ax.grid(ls='-', c='b', lw=0.3, alpha=0.05, which='minor')
    fig.set_size_inches(8,6)
    fig.tight_layout()
    fig.savefig("{}-{}-{}-{}-comparison.pdf".format("alldata", f1, f2, wav))


def extract_filter_id(fn):
    if fn.startswith('FQ'):
        return fn[2:5]
    elif fn.startswith('F'):
        return fn[1:4]
    else:
        raise ValueError()


def fmt_ylabel(f1, f2, simple, latex=True):
    """Format the label for the y-axis"""
    if latex:
        j1 = extract_filter_id(f1)
        j2 = extract_filter_id(f2)
        ylabel = r'Filter ratio: $[R_{{{0}}} \, / \, (\widetilde{{k}}_{{{0},{1}}} \, R_{{{1}}})]$'.format(j1, j2)
        extra = r'$ {{}}\, - \, r_0 S_{{{0}}}$'.format(j1)
    else:
        ylabel = 'Filter ratio: {} / (k {})'.format(f1, f2)
        extra = ' - r0 S'
    if not simple:
        ylabel += extra
    return ylabel


@arg("f1", default="FQ575N", help="Narrow-band filter", type=str)
@arg("f2", default="F547M", help="Broad-band filter", type=str)
@arg("--colorstrategy", help="Which version of continuum color to use",
     choices=["local", "global", "average"])
@arg("--fixedcolor", help="Set the k-value by hand - do not read from table", type=float)
@arg("--simple", help="Do not correct for line contamination of narrow filter")
@arg("--ymax", help="Set the plot limit of y-axis by hand", type=float)
def main(f1, f2, colorstrategy="global", fixedcolor=None, simple=False, ymax=None, alpha=0.6): 
    """Compare the intensity in two different filters - image versus spectra"""
    print("Filters are", f1, "and", f2)
    datasets = ["odh", "manu", "ring"]
    tabs = {}
    
    for dataset in datasets:
        if dataset == "ring":
            datapath = Path("../../RingNebula/WFC3/2013-Geometry")
            fn = "compare_{}_{}.tab".format(f1, f2)
        else:
            datapath = Path(".")
            fn = "{}_compare_{}_{}.tab".format(dataset, f1, f2)
        tabs[dataset] = Table.read(str(datapath / fn), format="ascii.tab", fill_values=('--', np.nan))

    plot_ew_ratio(tabs, wavtargets[f1], f1, f2,
                  colorstrategy=colorstrategy, fixedcolor=fixedcolor, simple=simple, ymax=ymax, alpha=alpha)

            
if __name__ == "__main__":
    # Plac is a  super-simple command line parser
    # http://plac.googlecode.com/hg/doc/plac.html
    argh.dispatch_command(main)

    # Argh is ANOTHER super-simple command line parser
    # https://pythonhosted.org/argh/tutorial.html
    # import argh; argh.dispatch_command(main)
