W547, W658, W575 = 169.0, 7.3, 4.7
Tn, Ta = 0.26, 0.23
k658 = 0.8  # We don't really know this one yet

def EWa(R575, k575=0.938, correction=1.2):
    return (W547/((k575/correction)*R575) - W575)/Ta

def EWn(R658):
    return (W547/(k658*R658) - W658)/Tn

def ratio_nii(R, R575, R658, k575=0.938, correction=1.2):
    """[N II] 5755/6584 ratio"""
    return R * (Tn/Ta) * (1.0 - W575*(k575/correction)*R575/W547) / (1.0 - W658*k658*R658/W547)

#def ratio_nii(R, R575, R658):
#    return R * (Tn/Ta) * (1.0 + W658/(Tn*EWn(R658))) / (1.0 + W575/(Ta*EWa(R575)))
