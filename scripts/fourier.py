# Copyright 2022 Tom Eulenfeld, MIT license

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import irfft
from matplotlib.ticker import ScalarFormatter
from qopen.source import source_model, moment_magnitude


Mws = r'$M_{\rm w}$'
M0s = r'$M_0$'
fcs = r'$f_{\rm c}$'
fc20 = r'$f_{\rm c2}$'
fc18 = r'$f_{\rm c1}$'
dotM = r'$\dot M$'
dotM20 = r'$\dot M_{2}$'
dotM18 = r'$\dot M_{1}$'
Ml20 = r'$M_{\rm l 2}$'
Ml18 = r'$M_{\rm l 1}$'

Mw = 0.75
M0 = moment_magnitude(Mw, inverse=True)
fc2 = 30
a=0.7
# fc1 = 42
fc1 = fc2 / a
N = 1000001
F = 2000
f = np.linspace(0, F//2, N)

f1 = 0.9
f2 = 200/0.9
mask = (f >= f1) & (f <= f2)
sds1 = source_model(f, M0, fc1, n=2, gamma=1)
sds2 = source_model(f, M0, fc2, n=2, gamma=1)

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.loglog(f[mask], sds1[mask], label='2018 (1)')
ax1.loglog(f[mask], sds2[mask], label='2020 (2)')

dt = 1 / F
T = 2 * (N - 1) * dt
t = np.linspace(-T//10, 9*T//10, 2 * (N -1))

M1 = np.roll(irfft(sds1), N // 5) / dt
M2 = np.roll(irfft(sds2), N // 5) / dt

ax1.set_xlim(f1, f2)
ax1.axvline(fc1, ls='--', color='C0', alpha=0.5)
ax1.axvline(fc2, ls='--', color='C1', alpha=0.5)
ax2.plot(t*1e3, M1)
ax2.plot(t*1e3, M2)
ax2.set_xlim(-25, 25)

ratio = np.max(M1) / np.max(M2)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0}$\times$10$^{{{1}}}$".format(base, int(exponent))
    else:
        return float_str

label = ('same moment magnitude\n'
         'smaller corner frequency\n'
         f'{Mws}={Mw}\n{M0s}={latex_float(M0)}Nm\n'
         f'a={fc20}/{fc18}=0.7')
ax1.annotate(label, (0.02, 0.2), xycoords='axes fraction')
label = (f'max({dotM20}) / max({dotM18}) = a\n'
         f'{Ml20} - {Ml18} = log$_{{10}}$(a) $\\approx$ $-$0.15\n'
         'smaller local magnitude')
ax2.annotate(label, (0.02, 0.98), xycoords='axes fraction', va='top')
ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.legend()
ax1.set_xlabel('frequency (Hz)')
ax1.set_ylabel(r'source displacement spectrum $\omega M$ (Nm)')
ax2.set_xlabel('time (ms)')
ax2.set_ylabel(f'moment rate function {dotM} (Nm/s)')
ax2.ticklabel_format(axis='y', useMathText=True)
ax2.set_ylim(None, 2.8e12)

y0 = 1.2e10
ap = dict(arrowstyle="->", connectionstyle="arc3,rad=1.5", shrinkA=2, shrinkB=0,)
ax1.annotate('', (fc2, y0), (fc1, y0), arrowprops=ap)
ax1.annotate(r'$\times$a', ((fc1*fc2)**0.5, y0*1.3), ha='center')

x0 = 0.3
ap = dict(arrowstyle="->", connectionstyle="arc3,rad=-0.5", shrinkA=2, shrinkB=0,)
ax2.annotate('', (x0, max(M2)), (x0, max(M1)), arrowprops=ap)
ax2.annotate(r'$\times$a', ((x0+3.8), (max(M2)*max(M1))**0.5), va='center')

t0 = 18
ap = dict(arrowstyle="->", connectionstyle="arc3,rad=-1.5", shrinkA=2, shrinkB=0,)
Mx = M2[t>t0/1e3][0]
t1 = t[M1>Mx][-1]*1e3
ax2.annotate('', (t0, Mx), (t1, Mx), arrowprops=ap)
ax2.annotate(r'$\times$1/a', ((t0+t1)/2, Mx+0.2e12), ha='center')

y0 = 0.8
ap = dict(arrowstyle="simple", connectionstyle="arc3,rad=0", shrinkA=0, shrinkB=0,facecolor='none')
ax1.annotate('', (0.53, y0), (0.47, y0), 'figure fraction', 'figure fraction',
             arrowprops=ap)
ax1.annotate(r'$\mathcal{F}^{-1}$', (0.5, y0 + 0.02), None, 'figure fraction', fontsize='xx-large',
             ha='center')
ax1.annotate('a)', (-0.15, 1), None, 'axes fraction', fontsize='large')
ax2.annotate('b)', (-0.15, 1), None, 'axes fraction', fontsize='large')

print(f'fc1={fc1:.2f}Hz')
print('label', label)
print('ratio fc {:.2f}'.format(fc1/fc2))
print('max of moment rate function: ratio, 1/ratio {:.2f} {:.2f}'.format(ratio, 1/ratio))
print('log10 of ratios {:.2f} {:.2f}'.format(np.log10(fc1/fc2), np.log10(ratio)))
print('M0 with trapz integration {:.2e} -> ratio of M0s {:.2f}'.format(np.trapz(M1, t), np.trapz(M1, t) / np.trapz(M2, t)))
fig.tight_layout()
fig.savefig('../figs/fourier.pdf')
