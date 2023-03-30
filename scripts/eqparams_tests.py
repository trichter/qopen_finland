# Copyright 2023 Tom Eulenfeld, MIT license

from copy import deepcopy
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import numpy as np
from qopen.source import calculate_source_properties, moment_magnitude

from sds_eqparams import calc_stressdrops, fc_vs_Mw, fc2stress_drop, plot_sds_wrapper

QOPENEVENTRESULTS = '../qopen/03_source/results.json'


def _secondary_yaxis_stress_drop(ax):
    M0 = moment_magnitude(0.9, inverse=True)
    funcs = (lambda fc: fc2stress_drop(fc, M0) / 1e6,
             lambda sd: fc2stress_drop(sd * 1e6, M0, inverse=True))
    ax2 = ax.secondary_yaxis('right', functions=funcs)
    ax2.set_ylabel(r'median stress drop $\Delta\sigma$ (MPa)')
    return ax2

def scale_events(results, R):
    for evres in results['events'].values():
        evres['W'] = [w/r if w is not None else None for w, r in zip(evres['W'], R)]

def _load_json(fname):
    with open(fname) as f:
        return json.load(f)

def _ld2dl(ld):
    return {k: [dic[k] for dic in ld] for k in ld[0]}

def test_different_n(plot_sds=False):
    oresults = _load_json(QOPENEVENTRESULTS)
    ld = []
    ns = (1.4, 1.5, 1.6, 1.7, 1.74, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3)
    for n in ns:
        print(f'n={n}')
        results = deepcopy(oresults)
        smo = {'n': n, 'gamma': 2, 'fc_lim': [5, 100], 'num_points': 5}
        print('  fit source params')
        calculate_source_properties(results,
                                    seismic_moment_method='robust_fit',
                                    seismic_moment_options=smo)
        print('  calc stress drops')
        sds = calc_stressdrops(results, oresults)
        print('  plot scaling')
        ld.append(fc_vs_Mw(sds, outlabel=f'_tests/eqparams_n={n:.2f}', tests=True))
        if plot_sds:
            print('  plot sds')
            plot_sds_wrapper(results, fname=f'../figs/eqparams_tests/sds_n={n:.2f}.pdf',
                             nx=8, figsize=(14, 50))
    dl = _ld2dl(ld)
    dl['n'] = ns
    with open('temp_n.json', 'w') as f:
        json.dump(dl, f)

def Rfbase(f, R2):
    return (f/f[0])**(np.log(R2)/np.log(f[-1]/f[0]))

def Rfexp(f, R2):
    return R2**((f-f[0])/(f[-1]-f[0]))

def test_different_Rf(plot_sds=False):
    oresults = _load_json(QOPENEVENTRESULTS)
    f = np.array(oresults['freq'])
    dd = {'Rfbase': [], 'Rfexp': []}
    R2s = [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
    for Rlabel, Rf in [('Rfbase', Rfbase), ('Rfexp', Rfexp)]:
        for R2 in R2s:
            print(f'{Rlabel}, R2 = {R2}')
            results = deepcopy(oresults)
            scale_events(results, Rf(f, R2))
            smo = {'n': None, 'gamma': 2, 'fc_lim': [5, 100], 'num_points': 5}
            print('  fit source params n=None')
            calculate_source_properties(results,
                                        seismic_moment_method='robust_fit',
                                        seismic_moment_options=smo)
            results2 = deepcopy(results)
            smo['n'] = 2
            print('  fit source params n=2')
            calculate_source_properties(results2,
                                        seismic_moment_method='robust_fit',
                                        seismic_moment_options=smo)
            print('  calc stress drops')
            sds = calc_stressdrops(results2, results)
            print('  plot scaling')
            d = fc_vs_Mw(sds, outlabel=f'_tests/eqparams_{Rlabel}_R2={R2}', tests=True)
            d['R2'] = R2
            dd[Rlabel].append(d)
            if plot_sds:
                print('  plot sds')
                plot_sds_wrapper(results2, fname=f'../figs/eqparams_tests/sds_{Rlabel}_R2={R2}.pdf',
                                 nx=8, figsize=(14, 50))
    dd['Rfbase'] = _ld2dl(dd['Rfbase'])
    dd['Rfexp'] = _ld2dl(dd['Rfexp'])
    with open('temp_Rf.json', 'w') as f:
        json.dump(dd, f)


def plotRf(ax=None):
    oresults = _load_json(QOPENEVENTRESULTS)
    f = np.array(oresults['freq'])
    if ax is None:
        ax = plt.subplot(111)
    ax.loglog(f, Rfbase(f, 10), 'C4', label=r'$R_f\sim f^x$ (linear)')
    ax.loglog(f, Rfexp(f, 10), 'C8', label=r'$R_f\sim y^f$ (exponential)')
    ax.legend(loc=(0.15, 0.72))
    ax.set_ylabel('frequency dependence of\nreference site amplification')
    ax.set_xlabel('frequency (Hz)')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.set_yticks([1, 10])
    ax.set_yticklabels(['1', r'$R_2$'])
    ax.set_xticks([3, 10, 100, 200])

AXLABELKW = dict(xy=(0, 1), xytext=(5, -5),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', size='large')


def plotntests():
    d = _load_json('temp_n.json')
    fig = plt.figure(figsize=(10, 3))
    ax1 = fig.add_subplot(121)
    ax3 = fig.add_subplot(122, sharex=ax1)
    ax1.plot(d['n'], d['medfc'], 'x-', color='0.2')
    ax3.axhline(-3, ls=':', color='k')
    ax3.plot(d['n'], d['slope'], 'x-', color='0.2', label='normal fit')
    ax3.plot(d['n'], d['slope_binned'], 'x-.', color='0.2', label='binned fit')
    ax3.set_ylim(-9.8, -2.9)
    ax1.set_xlabel(r'high-frequency falloff $n$')
    ax3.set_xlabel(r'high-frequency falloff $n$')
    ax1.set_ylabel(r'median corner frequency $f_{\rm c}$ (Hz)')
    ax3.set_ylabel('slope (scaling exponent)')
    ax1.set_yscale('log')
    ax2 = _secondary_yaxis_stress_drop(ax1)
    ax1.yaxis.set_minor_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax2.set_yticks([0.6, 1, 3, 6])
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax1.annotate('a)', **AXLABELKW)
    ax3.annotate('b)', **AXLABELKW)
    ax3.legend(loc=(0.15, 0.72))
    fig.tight_layout()
    fig.savefig('../figs/testn.pdf')
    return ax3.get_legend_handles_labels()


FF = FuncFormatter(lambda x, pos: x if x < 1 else int(round(x)))
def plotRftests(legend_handlers=None):
    d = _load_json('temp_Rf.json')
    R2 = d['Rfbase']['R2']
    fig = plt.figure(figsize=(10, 6))
    ax5 = fig.add_subplot(221)
    plotRf(ax5)
    ax1 = fig.add_subplot(223)
    ax3 = fig.add_subplot(224, sharex=ax1)
    ax4 = fig.add_subplot(222, sharex=ax1)
    ax1.plot(R2, d['Rfbase']['medfc'], 'x-C4')
    ax1.plot(R2, d['Rfexp']['medfc'], 'x-C8')
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax1.xaxis.set_major_formatter(FF)
    ax1.yaxis.set_minor_formatter(ScalarFormatter())
    ax1.yaxis.set_major_formatter(ScalarFormatter())
    ax2 = _secondary_yaxis_stress_drop(ax1)
    ax2.set_yticks([0.2, 1, 10])
    ax2.yaxis.set_major_formatter(FF)

    ax4.plot(R2, d['Rfbase']['medn'], 'x-C4')
    ax4.plot(R2, d['Rfexp']['medn'], 'x-C8')
    ax3.axhline(-3, ls=':', color='k')
    ax3.plot(R2, d['Rfbase']['slope'], 'x-C4', label='normal fit')
    ax3.plot(R2, d['Rfbase']['slope_binned'], 'x-.C4', label='binned fit')
    ax3.plot(R2, d['Rfexp']['slope'], 'x-C8')
    ax3.plot(R2, d['Rfexp']['slope_binned'], 'x-.C8')
    ax4.tick_params(axis='y')
    ax1.set_xlabel(r'$R_2$')
    ax3.set_xlabel(r'$R_2$')
    ax4.set_xlabel(r'$R_2$')
    ax1.set_ylabel(r'median corner frequency $f_{\rm c}$ (Hz)')
    ax4.set_ylabel(r'median high-frequency falloff $n$')
    ax3.set_ylabel('slope (scaling exponent)')
    ax3.set_ylim(-9.8, -2.9)
    ax5.annotate('a)', **AXLABELKW)
    ax4.annotate('b)', **AXLABELKW)
    ax1.annotate('c)', **AXLABELKW)
    ax3.annotate('d)', **AXLABELKW)
    if legend_handlers:
        ax3.legend(*legend_handlers, loc=(0.15, 0.72))
    fig.tight_layout()
    fig.savefig('../figs/testRf.pdf')


if __name__ == '__main__':
    test_different_n()
    test_different_Rf()
    legend_handlers = plotntests()
    plotRftests(legend_handlers=legend_handlers)
