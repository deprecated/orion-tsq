import numpy as np
from astropy.table import Table
import argh
from argh.decorators import arg
from matplotlib import pyplot as plt

@arg("--rmax", type=float)
@arg("--rmin", type=float)
@arg("--emax", type=float)
@arg("--emin", type=float)
def main(Eline="E5755", f1="FQ575N", f2="F547M", rmin=0.0,
         rmax=None, emin=0.0, emax=None):
    """Plot side-by-side maps of the filter ratio and EW"""
    tab = Table.read("manu_compare_{}_{}.tab".format(f1, f2), 
                     format="ascii.tab", fill_values=('--', 0.0))
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    m = np.logical_and(*[~tab[c].mask & np.isfinite(tab[c]) for c in (Eline, f1, f2)])

    x = tab['x'][m]
    y = tab['y'][m]
    Ratio = tab[f1][m]/tab[f2][m]
    EW = tab[Eline][m]
    sweetmask = np.ones_like(x).astype(bool)
    theta = np.radians(56.0)
    s = -x*np.cos(theta) + y*np.sin(theta)
    sweetmask[s > 10.0] = False
    sweetmask[s < -70.0] = False
    theta = np.radians(-34.0)
    s = -x*np.cos(theta) + y*np.sin(theta)
    sweetmask[s > 82.0] = False
    sweetmask[s < 40.0] = False
    size = np.where(sweetmask, 250.0, 50.0)
    scat0 = axes[0].scatter(x, y, c=EW, s=size, alpha=0.6,
                            vmin=emin, vmax=emax, cmap=plt.cmap.gray)
    fig.colorbar(scat0, ax=axes[0])
    axes[0].set_xlim(0.0, -90.0)
    axes[0].set_ylim(-90.0, 0.0)
    axes[0].axis('equal')
    axes[0].set_title(Eline)

    scat1 = axes[1].scatter(x, y, c=Ratio, s=size, alpha=0.6, vmin=rmin, vmax=rmax)
    fig.colorbar(scat1, ax=axes[1])
    axes[1].axis('equal')
    axes[1].set_title(f1 + '/' + f2)

    fig.set_size_inches(18, 9)
    fig.savefig("manu-compare-maps-{}-{}.pdf".format(f1, f2))


if __name__ == "__main__":
    argh.dispatch_command(main)
