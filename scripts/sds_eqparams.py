# Copyright 2023 Tom Eulenfeld, MIT license

import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress, median_abs_deviation as mad
from qopen.imaging import plot_all_sds, _secondary_yaxis_seismic_moment
from qopen.source import moment_magnitude
from qopen.util import linear_fit


Mls = r'$M_{\rm{L}}$'
Mws = r'$M_{\rm w}$'
fcs = r'$f_{\rm c}$'

QOPENEVENTRESULTS = '../qopen/03_source/results.json'
QOPENEVENTRESULTS2 = '../qopen/04_source_nconst/results.json'

QOPENEVENTRESULTS3 = '../qopen/06_source_2020/results.json'
QOPENEVENTRESULTS4 = '../qopen/07_source_2020_nconst/results.json'
FIXED_N = True

QOPENSITERESULTS = '../qopen/02_sites/results.json'

EQ_PARAMS = '../data/eq_params_{}.csv'
EPHEADER = 'evid,Ml,Mw,fc,n,stressdrop_MPa'
EPDTYPE = '<U23,f4,f4,f4,f4,f4'
EPFMT = '%s,%.3f,%.3f,%.3f,%.3f,%.3f'


def _linear_fit_L1(y, x, m0, b0):
    def cost_function(params, x, y):
        m, b = params
        return np.sum(np.abs(y - m * x - b))
    # to improve this, we have to use an algorithm finding global maxima
    out = minimize(cost_function, (m0, b0), args=(x, y))
    return out.x


def _load_json(fname):
    with open(fname) as f:
        return json.load(f)


def plot_sds_wrapper(results, select=None, **kw):
    def _shorten_eventids(events):
        for evid in list(events):
            events[evid[:13]] = events.pop(evid)

    annotate_label=(r'$M_{{\rm{{L}}}}$={Mcat:.1f} $M_{{\rm w}}$={Mw:.1f}' +
                    '\n' + r'$f_{{\rm c}}$={fc:.1f} Hz')
    if select:
        results['events'] = dict(list(results['events'].items())[:select])
    _shorten_eventids(results['events'])
    plot_all_sds(results, annotate=True, annotate_label=annotate_label, **kw)


def plot_sds_in_one():
    fixn = True
    results = _load_json(QOPENEVENTRESULTS2 if fixn else QOPENEVENTRESULTS)  # 2018
    results2 = _load_json(QOPENEVENTRESULTS4 if fixn else QOPENEVENTRESULTS3)  # 2020
     # results['events'] = dict(list(results['events'].items())[:23])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    freq = np.array(results['freq'])
    sdsa = [r['sds'] for r in results['events'].values()]
    sdsa2 = [r['sds'] for r in results2['events'].values()]
    ax.loglog(freq, np.transpose(sdsa), color='C0', alpha=0.2)
    ax.loglog(freq, np.transpose(sdsa2), color='C1', alpha=0.5)

    m, b =  -3.8705377975666306, 6.9800162592561605
    m6, b6 =  -5.075415912097096, 8.24289668988976

    fclim = np.array([15, 50, 70])
    ax.plot(fclim, moment_magnitude(np.log10(fclim)*m+b, inverse=True),
             color='C0')
    ax.plot(fclim, moment_magnitude(np.log10(fclim)*m6+b6, inverse=True),
             color='C1')
    ax.set_xlabel('frequency (Hz)')
    ax.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200])
    ax.set_xticklabels(['10', '20', '30', '40', '', '60', '', '80', '', '100', '200'])
    ax.set_ylabel(r'source displacement spectrum $\omega M$ (Nm)')
    ax.set_ylim(1e9/1.1, 1.1e12)

    fig.savefig('../figs/all_sds_in_one.pdf', bbox_inches='tight')


def calc_stressdrops(fname, fnamen=None, label=None):
    results = _load_json(fname) if isinstance(fname, str) else fname
    vals = [(evid, evres['Mcat'], evres['Mw'], evres['fc']) for evid, evres in results['events'].items() if 'Mw' in evres]
    evid, Ml, Mw, fc = map(np.array, zip(*vals))
    # load n values
    results = _load_json(fnamen) if isinstance(fnamen, str) else results if fnamen is None else fnamen
    nd = {evid: evres['n'] for evid, evres in results['events'].items() if 'Mw' in evres}
    n = np.array([nd.get(id_) for id_ in evid], dtype=float)
    print(f'high frequency fall-of mean: {np.nanmean(n):.3f}  '
          f'median: {np.nanmedian(n):.3f}  std: {np.nanstd(n):.3f}')
    sd = fc2stress_drop(fc, moment_magnitude(Mw, inverse=True)) / 1e6  # MPa
    Mwl = 0.8
    print('stress drop in MPa 2018')
    print('mean+-var sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.mean(sd), np.var(sd), 100 * np.var(sd)/np.mean(sd)))
    print('median+-mad sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.median(sd), mad(sd), 100 * mad(sd)/np.median(sd)))
    print('median+-geometric mad sd  {:.1f} -{:.1f} +{:.1f} ({:.0f}%, {:.0f}%)'.format(np.median(sd), geomad(sd)[0], geomad(sd)[1], 100 * geomad(sd)[0]/np.median(sd), 100 * geomad(sd)[1]/np.median(sd)))
    print('median sd Ml>', np.median(sd[Mw>=Mwl]))
    arr = np.asarray(list(zip(*[evid, Ml, Mw, fc, n, sd])), dtype={'names': EPHEADER.split(','), 'formats': EPDTYPE.split(',')})
    if label is not None:
        fname = EQ_PARAMS.format(label)
        np.savetxt(fname, arr, fmt=EPFMT, header=EPHEADER)
    return arr


def compare_mags():
    t1 = np.genfromtxt(EQ_PARAMS.format(2018), dtype=EPDTYPE, names=True, delimiter=',')
    t2 = np.genfromtxt(EQ_PARAMS.format(2020), dtype=EPDTYPE, names=True, delimiter=',')
    Ml = t1['Ml']
    Mw = t1['Mw']
    Ml2 = t2['Ml']
    Mw2 = t2['Mw']
    mmin, mmax = 0, np.max(Ml)
    m = np.linspace(mmin, mmax, 100)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, aspect='equal')

    # Kim, Wahlströhm, Uski 1989
    ak = 1.01 * 2 / 3
    bk = (16.93 - 7) * 2 / 3 - 6.07
    label = f'{Mws}={ak:.2f}{Mls} {bk:+.2f} (Kim et al. 1989)'
    ax.plot(m, ak * m + bk, '--', color='0.5', label=label, alpha=0.6)
    # Uski, Lund, Oinonen 2015
    # found in Kwiatek2019
    au = 1 / 0.83 * 2 / 3
    bu = 7.98 / 0.83 * 2 / 3 - 6.07
    label = f'{Mws}={au:.2f}{Mls} {bu:+.2f} (Lund et al. 2015)'
    ax.plot(m, au * m + bu, '-', color='0.5', label=label, alpha=0.6)

    a, b = linear_fit(Mw, Ml)
    label = f'{Mws}={a:.2f}{Mls} {b:+.2f}'
    print(label)
    ax.plot(m, a * m + b, '-C0', label=label, alpha=0.6)
    a, b = linear_fit(Ml[Mw>=0.6], Mw[Mw>=0.6])
    label = f'{Mls} = {a:.2f}{Mws} {b:+.2f}\n{Mws} = {1/a:.2f}{Mls} {-b/a:+.2f}'
    print(label)

    a2, b2 = linear_fit(Mw2, Ml2)
    label = f'{Mws}={a2:.2f}{Mls} {b2:+.2f}'
    print(label)
    ax.plot(m, a2 * m + b2, '-C1', label=label, alpha=0.6)

    ax.plot(Ml, Mw, 'x', ms=5, label='2018')
    ax.plot(Ml2, Mw2, 'o', ms=4, mfc='None', label='2020')

    # ax.plot(m, m / a - b / a, '--k', label=label, alpha=0.5)
    ax.legend(loc='upper left', frameon=False)
    _secondary_yaxis_seismic_moment(ax)
    ax.set_ylabel(f'moment magnitude {Mws}')
    ax.set_xlabel(r'local magnitude  $M_{\rm l}$')
    fig.savefig('../figs/mags2.pdf', bbox_inches='tight')


def fc_vs_Mw_vs_n():
    t1 = np.genfromtxt(EQ_PARAMS.format(2018), dtype=EPDTYPE, names=True, delimiter=',')
    Mw, fc, n = t1['Mw'], t1['fc'], t1['n']
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(131)
    ax1.plot(Mw, fc, 'x')
    ax1.set_xlabel(f'moment magnitude {Mws}')
    ax1.set_ylabel(f'corner frequency {fcs} (Hz)')
    ax1.set_yscale('log')
    ax1.set_yticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ax1.set_yticklabels(['', '20', '30', '40', '', '60', '', '', '', ''])
    ax2 = fig.add_subplot(132)
    ax2.hist(n, bins=np.arange(1.525, 2.26, 0.05), rwidth=0.9)
    ax2.set_xlabel('high frequency fall-off $n$')
    ax3 = fig.add_subplot(133)
    ax3.plot(Mw, n, 'x')
    ax3.set_xlabel(f'moment magnitude {Mws}')
    ax3.set_ylabel('high frequency fall-off $n$')
    ax1.set_ylim(15, 70)
    ax1.set_xlim(0.33, 1.96)
    plt.tight_layout()
    fig.savefig('../figs/eqparams2.pdf', bbox_inches='tight', pad_inches=0.1)


def fc2Mw(stress_drop, fc):
    r = 3500 * 0.21 / np.array(fc)  # Madariaga (1976) for S waves
    M0 = 16 * r ** 3 * stress_drop / 7
    return moment_magnitude(M0)


def fc2stress_drop(fc, M0, inverse=False):
    if inverse:
        stress_drop = np.array(fc)
        r = (7 * M0 / ( 16 * stress_drop)) ** (1/3)
        fc = 3500 * 0.21 / r
        return fc
    else:
        r = 3500 * 0.21 / np.array(fc)
        stress_drop = 7 * M0 / (16 * r ** 3)
        return stress_drop


def geomad(vals):
    lmad = mad(np.log(vals))
    med = np.median(vals)
    return med-med*np.exp(-lmad), med*np.exp(lmad)-med


def _logregr(Mw, fc, method='L2'):
    if method == 'L2':
        m2, b2, _, _, m2_stderr = linregress(Mw, np.log10(fc))
        m, b = 1 / m2, -b2 / m2
        m_stderr = m2_stderr / m2 ** 2
        print(f'L2 fit, Mw independent variable: M0 ∝ fc^{1.5*m:.2f}+-{1.5*m_stderr:.2f}')
        label = f'$M_0$ ∝ {fcs}$^{{{1.5*m:.2f} \pm {1.5*m_stderr:.2f}}}$'
        return m, b, m_stderr, label
    elif method == 'robust':
        m2, b2 = linear_fit(np.log10(fc), Mw, method='robust')
        m, b = 1 / m2, -b2 / m2
        print(f'L2 fit, Mw independent variable: M0 ∝ fc^{1.5*m:.2f}')
        label = f'$M_0$ ∝ {fcs}$^{{{1.5*m:.2f}}}$'
        return m, b, np.nan, label


def fc_vs_Mw(fname1, fname2=None, outlabel='', tests=False):
    dxxx = 0.26
    t1 = np.genfromtxt(fname1, dtype=EPDTYPE, names=True, delimiter=',') if isinstance(fname1, str) else fname1
    Mw1, fc1, sd1, n1 = t1['Mw'], t1['fc'], t1['stressdrop_MPa'], t1['n']
    if fname2:
        t2 = np.genfromtxt(fname2, dtype=EPDTYPE, names=True, delimiter=',')  if isinstance(fname2, str) else fname2
        Mw2, fc2, sd2, n2 = t2['Mw'], t2['fc'], t2['stressdrop_MPa'], t2['n']
    if tests:
        # We need to exclude some events for which the source model fitting did
        # not work, especially for the exponential Rf model with high R2
        # 5 Hz is the lower bound of fc in the optimization
        Mw1 = Mw1[fc1>5.1]
        sd1 = sd1[fc1>5.1]
        n1 = n1[fc1>5.1]
        fc1 = fc1[fc1>5.1]

    fig = plt.figure(figsize=(10, 5))
    box1 = [0.45, 0.1, 0.45, 0.85]
    box4 = [0.45, 0.98, 0.45, 0.18]
    box3 = [0.05, 0.8+dxxx, 0.3, 0.23]
    box5 = [0.05, 0.45+dxxx, 0.3, 0.23]
    box6 = [0.05, 0.1, 0.3, 0.23]
    box7 = [0.05, 0.1+dxxx, 0.3, 0.23]
    akwargs = dict(xy=(0, 1), xytext=(5, -5),
                   xycoords='axes fraction', textcoords='offset points',
                   va='top', size='large')

    # main axes f) Mw vs fc
    ax1 = fig.add_axes(box1)
    ax1.plot(fc1, Mw1, 'x', color='C0', label='2018')
    if fname2:
        ax1.plot(fc2, Mw2, 'o', color='C1', mfc='None', ms=4, label='2020')

    m, b, _, _, m_stderr = linregress(np.log10(fc1), Mw1)
    print(f'L2 fit, fc independent variable: M0 ∝ fc^{1.5*m:.2f}+-{1.5*m_stderr:.2f}')
    m, b = _linear_fit_L1(Mw1, np.log10(fc1), m, b)
    print(f'L1 fit, fc independent variable: M0 ∝ fc^{1.5*m:.2f}')

    m2, b2 = _linear_fit_L1(np.log10(fc1), Mw1, 1/m, -b/m)
    m, b = 1 / m2, -b2 / m2
    print(f'L1 fit, Mw independent variable: M0 ∝ fc^{1.5*m:.2f}')

    kw = dict(rotation=-35, rotation_mode='anchor',
              xytext=(0, -15), textcoords='offset points')
    ax1.annotate('0.1 MPa', (20, fc2Mw(0.1e6, 20)), **kw)
    ax1.annotate('1 MPa', (20, fc2Mw(1e6, 20)), **kw)
    ax1.annotate('10 MPa', (50, fc2Mw(10e6, 50)), **kw)
    fcx = 10 ** (moment_magnitude(1e13) / m - b / m)
    sdx = fc2stress_drop(fcx, 1e13) / 1e6
    print(f'fc={fcx:.1f}Hz and stress drop {sdx:.1f} Mpa for M0=1e13 Nm')
    ax1.set_ylabel(f'moment magnitude {Mws}')
    ax1.set_xlabel(f'corner frequency {fcs} (Hz)')

    # b) corner frequency histogram
    ax4 = fig.add_axes(box4, sharex=ax1)
    #bins = np.logspace(np.log10(15), np.log10(70), 41)
    bins = np.logspace(np.log10(1), np.log10(100), 120)
    if fname2:
        ax4.hist((fc2, fc1), bins=bins, histtype='barstacked', color=('C1', 'C0'), rwidth=0.9, zorder=10)
    else:
        ax4.hist(fc1, bins=bins, rwidth=0.9, zorder=10)
    ax4.tick_params(labelbottom=False)#, right=False, top=False)
    ax4.axvline(np.median(fc1), color='C0', ls='--')
    print(f'median fc1 {np.median(fc1):.2f}Hz')
    if fname2:
        ax4.axvline(np.median(fc2), color='C1', ls='--')
        print(f'median fc2 {np.median(fc2):.2f}Hz')
    if not tests:
        ax4.set_ylim(None, 18)

    # a) n histogram
    ax3 = fig.add_axes(box3)
    bins = np.arange(1.325, 2.31, 0.05) if not tests else 21
    if fname2:
        ax3.hist((n2, n1), bins=bins, histtype='barstacked', rwidth=0.9, color=('C1', 'C0'), zorder=10)
        ax3.axvline(np.median(n2), color='C1', ls='--')
    else:
        ax3.hist(n1, bins=bins, rwidth=0.9, zorder=10)
    ax3.axvline(np.nanmedian(n1), color='C0', ls='--')

    ax3.set_ylabel('counts')
    ax3.set_xlabel('high frequency fall-off $n$')
    ax1.annotate('f)', **akwargs)
    ax3.annotate('a)', **akwargs)
    if ax4:
        ax4.set_ylabel('counts')
        ax4.annotate('b)', **akwargs)

    # c) stress drop histogram
    bins = 10 ** np.arange(-1.05, 1.46, 0.1)
    ax5 = fig.add_axes(box5)
    ax5.axvline(np.median(sd1), color='C0', ls='--')
    print(f'median sd1 {np.median(sd1):.2f}MPa')

    if fname2:
        ax5.axvline(np.median(sd2), color='C1', ls='--')
        print(f'median sd2 {np.median(sd2):.2f}MPa')
        ax5.hist((sd2, sd1), bins=bins, histtype='barstacked', color=('C1', 'C0'), rwidth=0.9, zorder=10)
    else:
        ax5.hist(sd1, bins=bins, rwidth=0.9, zorder=10)
    if not tests:
        ax5.set_ylim(0, 50)

    ax5.set_xscale('log')
    ax5.set_xlabel('stress drop $\Delta\sigma$ (MPa)')
    ax5.set_ylabel('counts')
    ax5.xaxis.set_major_formatter(ScalarFormatter())
    ax5.annotate('c)', **akwargs)

    # e) stress drop vs binned Mw
    ax6 = fig.add_axes(box6)
    Mwmid = np.arange(0.5, 1.91, 0.2)
    sdv1 = [sd1[np.logical_and(mwmidv-0.1 <= Mw1, Mw1 < mwmidv+0.1)] for mwmidv in  Mwmid]
    sdm1 = np.array(list(map(np.median, sdv1)))
    sderr1 = np.transpose(list(map(geomad, sdv1)))
    ax6.errorbar(Mwmid-0.01, sdm1, yerr=sderr1, marker='x', ls='', label='2018')
    if fname2:
        sdv2 = [sd2[np.logical_and(mwmidv-0.1 <= Mw2, Mw2 < mwmidv+0.1)] for mwmidv in  Mwmid]
        sdm2 = np.array(list(map(np.median, sdv2)))
        sderr2 = np.transpose(list(map(geomad, sdv2)))
        ax6.errorbar(Mwmid+0.01, sdm2, yerr=sderr2, marker='o', ls='', ms=4, mfc='None', label='2020')
        ratio = sdm1[:len(sdm2)] / sdm2
        print('radom stress drop ratios',  round(np.nanmedian(ratio), 2))
    ax6.set_yscale('log')
    if not tests:
        ax6.set_ylim(0.3/1.1, 10*1.1)
    ax6.set_yticks([1, 10])
    ax6.yaxis.set_major_formatter(ScalarFormatter())

    ax6.set_ylabel('stress drop\n$\Delta\sigma$ (MPa)')
    ax6.set_xlabel(f'moment magnitude {Mws}')
    ax6.annotate('e)', **akwargs)
    del akwargs['size']
    ax6.annotate('         median and MAD', **akwargs)
    ax6.set_xticks(Mwmid)

    @matplotlib.ticker.FuncFormatter
    def myformatter(x, pos):
        return f'{x:.1f}' if 0.25 <= x <= 0.35 else ''
    ax6.yaxis.set_minor_formatter(myformatter)

    # d) fc vs binned Mw
    fcv1 = [fc1[np.logical_and(mwmidv-0.1 <= Mw1, Mw1 < mwmidv+0.1)] for mwmidv in  Mwmid]
    fcm1 = np.array(list(map(np.median, fcv1)))
    fcerr1 = np.transpose(list(map(geomad, fcv1)))
    ax7 = fig.add_axes(box7, sharex=ax6)
    ax7.errorbar(Mwmid-0.01, fcm1, yerr=fcerr1, marker='x', ls='')
    if fname2:
        fcv2 = [fc2[np.logical_and(mwmidv-0.1 <= Mw2, Mw2 < mwmidv+0.1)] for mwmidv in  Mwmid]
        fcm2 = np.array(list(map(np.median, fcv2)))
        fcerr2 = np.transpose(list(map(geomad, fcv2)))
        ax7.errorbar(Mwmid+0.01, fcm2, yerr=fcerr2, marker='o', ms=4, mfc='None', ls='')
        ratio = fcm1[:len(fcm2)] / fcm2
        print('radom corner freq ratio', round(np.nanmedian(ratio), 2))
    ax7.set_yscale('log')
    ax7.yaxis.set_major_formatter(ScalarFormatter())
    if tests:
        ax7.yaxis.set_minor_formatter(ScalarFormatter())
    ax7.tick_params(labelbottom=False)
    ax7.set_ylabel(f'corner frequency\n{fcs} (Hz)')
    ax7.annotate('d)', **akwargs)
    ax7.annotate('         median and MAD', **akwargs)
    if not tests:
        ax7.set_yticks([20, 30, 40, 50])
    ax7.set_xlim(0.35, 2.05)

    # line plots for ax1 and ax7
    fclim = np.array([5, 120])
    #method = 'robust' if tests else 'L2'
    method = 'L2'
    m, b, m_stderr, label = _logregr(Mw1, fc1, method=method)
    slope = 1.5*m
    print('m, b = ', m, b)
    ax1.plot(fclim, np.log10(fclim)*m+b, zorder=-1, label=label, color='C0')

    mwlim = np.array((0.3, 2))
    m, b, m_stderr, label = _logregr(Mwmid, fcm1, method=method)
    slope_binned = 1.5*m
    ax7.plot(mwlim, 10**((mwlim-b)/m), '-.', zorder=-1, label=label, color='C0', alpha=0.5)
    ax1.plot(10**((mwlim-b)/m), mwlim, '-.', zorder=-1, label=label, color='C0', alpha=0.5)
    if fname2:
        m, b, m_stderr, label = _logregr(Mw2, fc2, method=method)
        ax1.plot(fclim, np.log10(fclim)*m+b, zorder=-1, label=label, color='C1')
        m, b, m_stderr, label = _logregr(Mwmid[~np.isnan(fcm2)], fcm2[~np.isnan(fcm2)], method=method)
        ax7.plot(mwlim, 10**((mwlim-b)/m), '-.', zorder=-1, label=label, color='C1', alpha=0.5)
        ax1.plot(10**((mwlim-b)/m), mwlim, '-.', zorder=-1, label=label, color='C1', alpha=0.5)

    kw = dict(ls='--', color='0.5', zorder=-1)
    ax1.plot(fclim, fc2Mw(0.1e6, fclim), **kw)
    ax1.plot(fclim, fc2Mw(1e6, fclim), label=f'$M_0$ ∝ {fcs}$^{{-3}}$', **kw)
    ax1.plot(fclim, fc2Mw(10e6, fclim), **kw)
    ax1.set_xscale('log')
    ax1.set_xticks([20, 30, 40, 50, 60, 70])
    if not tests:
        ax1.set_xlim(18, 70)
    else:
        ax1.set_xlim(np.min(fc1)/1.1, np.max(fc1)*1.1)
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_ylim(0.33, 1.96)
    ax1.tick_params(top=False)
    _secondary_yaxis_seismic_moment(ax1)
    ax1.legend()

    if outlabel is not None:
        fname = '../figs/eqparams{}.pdf'.format(outlabel)
        fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)
    return {'medn': float(np.nanmedian(n1)),
            'medfc': float(np.median(fc1)),
            'medsd': float(np.median(sd1)),
            'slope': slope,
            'slope_binned': slope_binned}


if __name__ == '__main__':
    kw = dict(nx=8, figsize=(12, 4))
    plot_sds_wrapper(QOPENEVENTRESULTS2, select=23, fname='../figs/some_sds.pdf', **kw)
    plot_sds_wrapper(QOPENEVENTRESULTS4, fname='../figs/sds_2020.pdf', **kw)
    plot_sds_in_one()

    calc_stressdrops(QOPENEVENTRESULTS2, QOPENEVENTRESULTS, '2018')
    calc_stressdrops(QOPENEVENTRESULTS4, QOPENEVENTRESULTS3, '2020')
    compare_mags()
    fc_vs_Mw_vs_n()
    fc_vs_Mw(EQ_PARAMS.format(2018), EQ_PARAMS.format(2020))

    sm1 = moment_magnitude(0.6, inverse=True)
    sm2 = moment_magnitude(0, inverse=True)
    fac = round(sm2 ** (-1/4.7) / sm1 ** (1/-4.7), 2)
    print(f'offset of 0.6 in magnitude -> factor {fac:.2f} in corner frequency')
