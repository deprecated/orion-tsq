# Central coordinates were originally taken from ODH paper
# Except S90 is adjusted to have same RA as the others
# UPDATE: 17 Apr 2014 - correct the RAs from comparing the profiles
# S30 should be shifted about 10'' = 0.67s to the E
# S60 and S90 we move by the relative offsets in their FITS headers: -0.19, -0.14
slit_center = {
    "S30": "5:35:17.17 -5:23:53",
    "S60": "5:35:16.98 -5:24:23",
    "S90": "5:35:17.03 -5:24:53",
}
slit_width = 1.9                # arcsec
# From Sec 2.1 of ODH 2010
# (confirmed from http://www.ctio.noao.edu/noao/content/15mRC-Camera)
pixel_size = 1.3               # arcsec 
# (taking slit length of 330 pix = 3 x 143 arcsec - from Tab 2 of ODH2010)
PA = -90                         # All slits are E-W
