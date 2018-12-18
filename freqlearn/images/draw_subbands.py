# coding: utf-8
import dtcwt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', size=18)
import numpy as np
from math import *
xfm = dtcwt.Transform2d(biort='near_sym_b', qshift='qshift_b')
H = np.zeros((2,6,512,512))
for j in range(2):
    for l in range(6):
        x = np.zeros((512,512))
        x[256,256] = 1
        p = xfm.forward(x, nlevels=2)
        p.lowpass = np.zeros_like(p.lowpass)
        g = np.zeros((6,2), dtype='complex')
        g[l,j] = 1
        y = xfm.inverse(p, gain_mask=g)
        Y = np.fft.fft2(y)
        H[j,l] = np.fft.fftshift(20*np.log10(np.abs(Y)/np.abs(Y).max()+1e-6))

x = np.zeros((512,512))
x[256,256] = 1
p = xfm.forward(x, nlevels=2)
g = np.zeros((6,2), dtype='complex')
y = xfm.inverse(p, gain_mask=g)
Y = np.fft.fft2(y)
Hl = np.fft.fftshift(20*np.log10(np.abs(Y)/np.abs(Y).max()+1e-6))
l = np.linspace(-np.pi, np.pi, 512)
l1 = np.linspace(0, np.pi, 256)
A, B = np.meshgrid(l, l1)
#  colors = ['xkcd:azure', 'xkcd:brown', 'xkcd:coral', 'xkcd:darkblue',
          #  'xckd:green', 'xkcd:pink', 'xkcd:teal']
colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
fig, ax = plt.subplots()
#  pos = [(.125, .125), (.125, .75), (.375, .75), (.625, .75), (.875, .75), (.875,
       #  .125)]
c1 = 1.1*5*pi/8
c2 = .9*3*pi/8
pos = [(-c1, c2), (-c1, c1), (-c2, c1),
       (c2, c1), (c1, c1), (c1, c2)]
for j in range(2):
    for k,l in enumerate([2,1,0,5,4,3]):
        cs = ax.contour(A,B,H[j,l,:256][::-1,::-1],colors=[colors[l]],levels=[-3, -1],
                        linestyles='solid')
        #  if j == 0:
            #  ax.clabel(cs, [-1, -3], fmt={-1: '-1dB', -3: '-3dB'})
for k,l in enumerate([2,1,0,5,4,3]):
    ax.text(pos[k][0], pos[k][1], "{}°".format(75-30*k), color=colors[5-l],
            fontsize=14, ha='center', va='center')
cs = ax.contour(A,B,Hl[:256][::-1,::-1],colors=[colors[-1]],levels=[-3, -1], linestyles='solid',
           label='lowpass')
#  ax.clabel(cs, fmt={-1: 'lowpass', -3:''})
ax.set_xlabel(r'Horizontal Frequency - $w_1$')
ax.set_ylabel(r'Vertical Frequency - $w_2$')
pi = np.pi
plt.xticks([-pi, -pi/2, -pi/4, 0, pi/4, pi/2, pi],
           [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$-\frac{\pi}{4}$', '$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\pi$'])
#  plt.yticks([-pi, -pi/2, -pi/4, 0, pi/4, pi/2, pi],
           #  [r'$-\pi$', r'$-\frac{\pi}{2}$', r'$\frac{\pi}{4}$', '$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.yticks([0, pi/4, pi/2, pi],
           ['$0$', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', r'$\pi$'])
plt.tight_layout()
#  plt.axis('equal')
#  plt.xlim(left=-np.pi, right=np.pi)
#  plt.ylim(bottom=0, top=np.pi)
plt.show()
