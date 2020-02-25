import torch
import numpy as np

# amax = 0
amin = 1
for a in np.linspace(-1,1,100):
    for b in np.linspace(-1,1,100):
        for c in np.linspace(-1, 1, 100):
            term1 = 1/4 + 1/4*(-a+b+c)
            term2 = 1-(np.arccos(a)+np.arccos(b)+np.arccos(c))/ (2*np.pi)
            if term1 > 0 and term2>0:
                ratio = term2 / term1
                if ratio <= amin:
                    amin = ratio

print('min', amin)
