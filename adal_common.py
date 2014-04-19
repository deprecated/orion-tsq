slit_width = 1.03              # arcsec
pixel_size = 1.2                # arcsec

slit_center = {
    5: "05:35:15.9 -05:23:50.3",
    6: "05:35:15.2 -05:23:53.1",
} 

slit_PA = {
    5: 50,
    6: 72,
}

# Fine-scale corrections to position along slit, due to fact that
# reference point does not seem to be exactly in the center of the
# slit
slit_xshift = {
    5: 2.5, 
    6: 3.0,
}

Bands = {
    "FQ436N": "blue", 
    "FQ575N": "red", 
    "FQ672N": "red", 
    "FQ674N": "red", 
    "F673N":  "red", 
    "F469N":  "blue", 
    "F487N":  "blue", 
    "F656N":  "red", 
    "F658N":  "red", 
    "F547M":  "red", 
    "F502N":  "blue", 
    "FQ437N": "blue", 
}
