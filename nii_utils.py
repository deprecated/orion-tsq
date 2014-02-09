W547, W658, W575 = 169.0, 7.3, 4.7
Tn, Ta = 0.26, 0.23
k658 = 0.8  # This is uncertain, but it doesn't affect things much

K0 = 14.632e-10 # calibration constant for theoretical case (multiply by T)

# Calibration constants from Ring photometry
K658fit = 3.95e-10
K575fit = 4.28e-10

def EWa(R575, k575=0.938, correction=1.0):
    return (W547/((k575/correction)*R575) - W575)/Ta

def EWn(R658):
    return (W547/(k658*R658) - W658)/Tn

def ratio_nii(R, R575, R658, k575=0.938, correction=1.0, Ktype="predicted"):
    """[N II] 5755/6584 ratio in ENERGY FLUX UNITS"""
    # First-order basic calibration with the K values
    if Ktype.lower().startswith("pred"):
        ratio = R*(Tn/Ta)*(6584.0/5755.0)
    elif Ktype.lower().startswith("fit"):
        ratio = R*(K575fit/K658fit)*(6584.0/5755.0)
    # Second-order correction for continuum contamination
    ratio *= (1.0 - W575*(k575/correction)*R575/W547) / (1.0 - W658*k658*R658/W547)
    return ratio

#def ratio_nii(R, R575, R658):
#    return R * (Tn/Ta) * (1.0 + W658/(Tn*EWn(R658))) / (1.0 + W575/(Ta*EWa(R575)))
