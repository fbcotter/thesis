# coding: utf-8
import matplotlib.pyplot as plt
import numpy as np
from math import *
import dtcwt
xfm = dtcwt.Transform2d('near_sym_b', 'qshift_b')
x = np.zeros((128,64))
p = xfm.forward(x, nlevels=2)
m = p.highpasses[1].shape[0] // 2
r = int(.8 * m)
fig, ax = plt.subplots()
w = np.array([-1j, 1j, -1j, -1, 1, -1], 'complex')
for l in range(6):
    if l < 3:
        theta = 15+30*l
    else:
        theta = 15+30*l - 180
    p.highpasses[1][int(m-r*sin(theta*pi/180)), int(r*cos(theta*pi/180)), l] = w[l]
y = xfm.inverse(p)
ax.imshow(y, cmap='gray')
m = y.shape[0] // 2
r = int(.88 * m)
for l in range(6):
    if l < 3:
        theta = 15+30*l
    else:
        theta = 15+30*l - 180
    y = int(m - r*sin(theta*pi/180))
    x = int(r*cos(theta*pi/180))
    plt.text(x,y,"{}{}".format(theta, r"$^{\circ}$"), color='b', fontsize=11)
plt.show()
