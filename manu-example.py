import numpy as np
import pylab as pl
import pyfits as pf

# Read the table
table = 'table_M42_sr.fits'
thdu = pf.open(table)
tdata = thdu[1].data
thdr = thdu[1].header
thdu.close()

# Conditions to select region and fibers
# Arcseconds relative to Ori C
# Example for rectangular region
rec_region = (-30, -30, -10, -10)      
ra_mask = (tdata['dRA'] < rec_region[2])*(tdata['dRA'] > rec_region[0])
dec_mask = (tdata['dDEC'] > rec_region[1])*(tdata['dDEC'] < rec_region[3])
mask = ra_mask*dec_mask

# Read the image
hdu = pf.open('M42_sr.fits')
data = hdu[0].data
hdr = hdu[0].header

region_spectra = data[mask]

"""
In this case, 80 spectra are selected!!!
