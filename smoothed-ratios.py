from astropy.io import fits
import argh
import glob

def main(prefix="full_", suffix="-s150"):
    """Take ratios of all the narrow filters with F547M"""
    f2 = fits.open(prefix + "F547M" + suffix + ".fits")
    for fname in glob.glob(prefix + "F*N" + suffix + ".fits"):
        f1 = fits.open(fname)
        f1[0].data /= f2[0].data
        f1.writeto(fname.replace(suffix, "_F547M" + suffix), clobber=True)

if __name__ == "__main__":
    argh.dispatch_command(main)
