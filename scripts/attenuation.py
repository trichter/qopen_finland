# Copyright 2020 Tom Eulenfeld, MIT license

import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from qopen.core import collect_results
from qopen.imaging import calc_dependent
from qopen.util import gerr
from qopen.util import linear_fit

from util import MyHandlerTuple

MARKERS = ["o","v","^","<",">","s","p","P","*","h","H","+","x",'.', "X","D","d","|","_","1","2","3","4","8"]
AXLABELKW = dict(xy=(0, 1), xytext=(-50, 5),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', size='x-large')


fin = '../qopen/01_go/results.json'
QDATA = '../data/Q.json'


def Q(q, obs='sc', v0=None, error=False):
    if v0 is None:
        v0 = np.mean([ev['v0'] for ev in q['events'].values()])
        #v0 = q['config']['v0']
    freq = np.array(q['freq'])
    if obs == 'sc':
        mean = np.array(q['g0']) * v0 / (2 * np.pi * freq)
    else:
        mean = np.array(q['b']) / (2 * np.pi * freq)
    if not error:
        return freq, mean
    q = collect_results(q)
    if obs == 'sc':
        vals = np.array(q['g0']) * v0 / (2 * np.pi * freq)
    else:
        vals = np.array(q['b']) / (2 * np.pi * freq)
    mean2, err1, err2 = gerr(vals, axis=0, robust=True)
    np.testing.assert_allclose(mean, mean2)
    return freq, mean, (err1, err2)


def fit_kappa(f, Q):
    a, b = linear_fit(np.log(Q), np.log(f))
    kappa = -0.5 * a
    print(f'slope {a:.2}, kappa parameter {kappa:.2f} using frequencies >={f[0]:.1f} Hz')


def printQ_json(freq, Qsc, Qi, label=None):
    freqx = np.round(freq, 3).tolist()
    Qscx = np.round(Qsc*1e3, 3).tolist()
    Qix = np.round(Qi*1e3, 3).tolist()
    print(f""""{label}": {{
              "f": {freqx},
              "Qsc": {Qscx},
              "Qi": {Qix}}}""")


def plotQ():
    with open(fin) as f:
        q1 = json.load(f)
    fig = plt.figure(figsize=(8, 5.5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, sharex=ax1, sharey=ax1)


    freq, Qsc = Q(q1, 'sc')
    _, Qi = Q(q1, 'i')
    fit_kappa(freq[freq>5], Qsc[freq>5])

    printQ_json(freq, Qsc, Qi, label='Eulenfeld2023')
    l1 = ax1.errorbar(*Q(q1, 'sc', error=True), color='C9', marker='.', ms=6, zorder=5)

    ax1.set_xlabel('frequency (Hz)')
    ax1.set_ylabel(r'scattering strength $Q_{\mathrm{sc}}^{-1}$')

    l2 = ax2.errorbar(*Q(q1, 'i', error=True), color='C6', marker='.', ms=6, zorder=5)

    with open(QDATA) as f:
        Q2 = json.load(f)
    Q2 = {qk: qv for qk, qv in Q2.items() if 'Eulenfeld2023' in qv.get('ref_in', [])}
    print('used references:')
    print(', '.join([k.split('_')[0] for k in Q2.keys()]))
    lines, labels  = [], []
    for i, (label, r) in enumerate(Q2.items()):
        ms = MARKERS[i]
        c = str(0.4 + 0.4 * i / len(Q2))
        label = r['citation'] + ', ' + r['region']
        line, = ax1.loglog(r['f'], np.array(r['Qsc'])*1e-3, color=c, marker=ms)#, label=label)
        ax2.loglog(r['f'], np.array(r['Qi'])*1e-3, color=c, marker=ms)
        labels.append(label)
        lines.append(line)
    ax2.set_xlabel('frequency (Hz)')
    ax2.set_ylabel(r'intrinsic attenuation $Q_{\mathrm{intr}}^{-1}$')
    ax1.set_xticks((1, 10, 100))
    ax1.set_xticklabels(('1', '10', '100'))
    ax2.set_xticks((1, 10, 100))
    ax2.set_xticklabels(('1', '10', '100'))

    fig.legend(lines + [(l2, l1)], labels + ['this study'],
               loc='lower center',
                ncol=2, mode="expand", fontsize='small',
                handler_map={tuple: MyHandlerTuple(mdivide=None)})
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.45)

    ax1.annotate('b)', **AXLABELKW)
    ax2.annotate('c)', **AXLABELKW)
    fig.savefig('../figs/Q.pdf')


def plotl(ax=None, marker=None):
    with open(fin) as f:
        result = json.load(f)
    fig = None
    if ax is None:
        fig = plt.figure(figsize=(5, 3))
        ax = fig.add_subplot(111)
    kw = {'marker': marker}
    if marker == 'x':
        kw['mew'] = 1
        kw['ms'] = 4
    freq = np.array(result['freq'])
    v0 = result['config']['v0']
    col = collect_results(result, only=('g0', 'b', 'error'))
    lsc_all = calc_dependent('lsc', col['g0'], freq=freq, v0=v0)
    li_all = calc_dependent('li', col['b'], freq=freq, v0=v0)
    lsc, err1, err2 = gerr(lsc_all, axis=0, robust=True)
    lscerrs = (err1, err2)
    li, err1, err2 = gerr(li_all, axis=0, robust=True)
    lierrs = (err1, err2)
    ax.errorbar(freq*1.02, lsc, color='C9', ls='-', yerr=lscerrs, label='transport\nmean free path', **kw)
    ax.errorbar(freq/1.02, li, color='C6', ls='-', yerr=lierrs, label='intrinsic\nabsorption length', **kw)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())

    ax.set_xlabel('frequency (Hz)')
    ax.set_ylabel('path length (km)')
    ax.set_xlim(1.95, 520)
    ax.set_xticks([2, 10, 100, 500])
    ax.set_ylim([10/1.1, None])
    ax.legend()
    fig.tight_layout()
    ax.annotate('a)', **AXLABELKW)
    if fig is not None:
        fig.savefig('../figs/l.pdf', bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    plotQ()
    plotl()
