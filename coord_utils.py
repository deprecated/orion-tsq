"""Utilities for creating coordinate grids based on FITS headers, and
extracting the signal in apertures

"""
import numpy as np
import astropy.coordinates as coord
import astropy.units as u
from astropy import wcs


def set_coord(coordstr):
    # return coord.ICRSCoordinates(coordstr=coordstr, unit=(u.hour, u.deg))
    return coord.ICRS(coordstr=coordstr, unit=(u.hour, u.deg))


def get_radec(hdr):
    """Return arrays of RA and DEC corresponding to a FITS header hdr"""
    w = wcs.WCS(hdr)
    wnx, wny = hdr["NAXIS1"], hdr["NAXIS2"]
    X, Y = np.meshgrid(np.arange(wnx), np.arange(wny))
    RA, DEC = w.all_pix2world(X, Y, 0)
    return RA, DEC


def slitxy_from_radec(RA, DEC, center="05:35:15.2 -05:23:53.1", PA=0.0):
    center = set_coord(center)
    RA0, DEC0 = center.ra.deg, center.dec.deg
    dRA = (RA - RA0)*3600*np.cos(np.radians(DEC0))
    dDEC = (DEC - DEC0)*3600
    COSPA, SINPA = np.cos(np.radians(PA)), np.sin(np.radians(PA))
    xslit = dRA*SINPA + dDEC*COSPA
    yslit = -dRA*COSPA + dDEC*SINPA
    return xslit, yslit


def lookup_th1C_coord_str():
    return coord.ICRS.from_name('tet01 ori c').to_string(sep=':')


def radec_offsets_from_slitx(xslit, center="05:35:15.2 -05:23:53.1",
                             origin='5:35:16.4638 -5:23:22.8486', PA=0.0):
    """Find (RA, Dec) offsets from origin for each point along a slit.

Converts position along slit `xslit` (in arcsec) into (RA, Dec) offset
(in arcsec) from `origin` (default: th1C).  Also requires slit `center`
and slit `PA` (in degrees E of N)
    """
    center = set_coord(center)
    origin = set_coord(origin)
    COSPA, SINPA = np.cos(np.radians(PA)), np.sin(np.radians(PA))
    # First find offsets wrt slit center
    dRA = xslit*SINPA
    dDEC = xslit*COSPA
    # Then adjust to be wrt origin
    dRA += center.ra.arcsec - origin.ra.arcsec
    dDEC += center.dec.arcsec - origin.dec.arcsec
    
    return dRA, dDEC
