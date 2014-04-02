import sys
import numpy as np
import matplotlib.pyplot as plt
import astropy.table
from astropy.table import Table
import argh
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
    

def plot_ew_ratio(tab, wav, f1, f2, dataset="odh",
                  fixedcolor=None,
                  colorstrategy="average", simple=False, logscale=False,
                  suffix="", ymax=None, alpha=0.6):
    """Make a plot of Filter Ratio versus equivalent width"""
    xo = tab['E'+wav]
    xe = 3*tab['dE'+wav]
    snscale = {"odh": 2.0, "manu": 0.1, "adal": 1.0}
    r0, q1, q2 = rqq(get_exact_wav(wav), wavf, filts[f1], filts[f2])

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
    if  np.any(~tab['Sum(E/W)_2'].mask):
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
    try:
        mask = ~tab[f1].mask  
    except AttributeError:
        mask = np.ones_like(xo).astype(bool)

    sm = find_sweetspot_mask(tab['x'], tab['y']) 
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(xo[sm], yo[sm], xerr=xe[sm], yerr=ye[sm], fmt=None, zorder=0, alpha=alpha)
    #scatter(xo, yo, c=bigtab[f2], s=2*snr, alpha=0.6, vmax=0.2)
    scat = ax.scatter(xo[sm], yo[sm], c=d[sm], s=snscale[dataset]*snr[sm], alpha=alpha)

    if logscale: 
        xmin = xo[mask].min()/1.1
    else:
        xmin = 0.0
    xmax = xo[mask].max()*1.1
    x = np.linspace(xmin, xmax, 300)

    ax.plot(x, prelaunch_ratio(x, r0, q1, q2), label="Pre-launch calibration")
    p = np.poly1d(np.polyfit(xo[mask], yo[mask], 1))
    y0 = p(x)
    ax.plot(x, y0, "r-", label="Linear fit")

    if logscale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(xmin, xmax)
    else:
        ax.set_ylim(0.0, ymax)
        ax.set_xlim(0.0, xmax)

    cbar = fig.colorbar(scat)
    cbar.set_label("Distance from star, arcsec")
    #cbar.set_label("F547M brightness")
    cbar.solids.set_edgecolor("face")

    ax.set_xlabel(r'Equivalent width: $E_{{{}}}$, Angstrom'.format(wav), fontsize='x-large')
    
    ylabel = fmt_ylabel(f1, f2, simple)
    ax.set_ylabel(ylabel, fontsize='x-large')
    ax.minorticks_on()
    ax.legend(loc="upper left")
    ax.grid(ls='-', c='b', lw=0.3, alpha=0.3)
    ax.grid(ls='-', c='b', lw=0.3, alpha=0.05, which='minor')
    fig.set_size_inches(8,6)
    fig.tight_layout()
    fig.savefig("{}-{}-{}-{}{}-comparison.pdf".format(dataset, f1, f2, wav, suffix))


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
@arg("--dataset", help="Which spectrophotometric data to use",
     choices=["odh", "adal", "manu"])
@arg("--colorstrategy", help="Which version of continuum color to use",
     choices=["local", "global", "average"])
@arg("--fixedcolor", help="Set the k-value by hand - do not read from table", type=float)
@arg("--simple", help="Do not correct for line contamination of narrow filter")
@arg("--logscale", help="Use logarithmic scales for x- and y-axes")
@arg("--pa", help="Restrict data to the slit with this PA", type=int)
@arg("--ymax", help="Set the plot limit of y-axis by hand", type=float)
def main(f1, f2, dataset="odh", colorstrategy="global", fixedcolor=None,
         simple=False, logscale=False, pa=None, ymax=None, alpha=0.6): 
    """Compare the intensity in two different filters - image versus spectra"""
    print("Filters are", f1, "and", f2)
    spectab = astropy.table.join(get_spectab(f1, dataset),
                                 get_spectab(f2, dataset)["Section", "Sum(E/W)", "dSum"],
                                 keys=["Section"], join_type="left")
    tab = astropy.table.join(get_filtertab(f1, f2, dataset), spectab, join_type="left")
    if 'PA' in tab.colnames:
        tab.sort(['PA', 'x0'])
    else:
        tab.sort(['x', 'y'])
    tab.write("{}_compare_{}_{}.tab".format(dataset, f1, f2), format="ascii.tab")

    if pa is not None:
        m = tab['PA'] == pa
        tab = tab[m]
        suffix = '-PA{}'.format(pa)
    else:
        suffix = ''

    plot_ew_ratio(tab, wavtargets[f1], f1, f2, dataset=dataset,
                  colorstrategy=colorstrategy, fixedcolor=fixedcolor, simple=simple,
                  logscale=logscale, suffix=suffix, ymax=ymax, alpha=alpha)

            
if __name__ == "__main__":
    # Plac is a  super-simple command line parser
    # http://plac.googlecode.com/hg/doc/plac.html
    argh.dispatch_command(main)

    # Argh is ANOTHER super-simple command line parser
    # https://pythonhosted.org/argh/tutorial.html
    # import argh; argh.dispatch_command(main)
