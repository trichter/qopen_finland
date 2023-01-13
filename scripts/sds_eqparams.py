# Copyright 2022 Tom Eulenfeld, MIT license

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
EPHEADER = 'evid,Ml,Mw,fc,n,stressdrop'
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


def plot_sds_wrapper():
    def _shorten_eventids(events):
        for evid in list(events):
            events[evid[:13]] = events.pop(evid)

    fixn = True
    results = _load_json(QOPENEVENTRESULTS2 if fixn else QOPENEVENTRESULTS)  # 2018
    results2 = _load_json(QOPENEVENTRESULTS4 if fixn else QOPENEVENTRESULTS3)  # 2020
    annotate_label=(r'$M_{{\rm{{L}}}}$={Mcat:.1f} $M_{{\rm w}}$={Mw:.1f}' +
                    '\n' + r'$f_{{\rm c}}$={fc:.1f} Hz')
    results['events'] = dict(list(results['events'].items())[:23])
    _shorten_eventids(results['events'])
    _shorten_eventids(results2['events'])
    plot_all_sds(results, fname='../figs/some_sds.pdf', nx=8, figsize=(12, 4),
                 annotate=True, annotate_label=annotate_label)
    plot_all_sds(results2, fname='../figs/sds_2020.pdf', nx=8, figsize=(12, 4),
                 annotate=True, annotate_label=annotate_label)


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


def calc_stressdrops(fixn=True):
    results = _load_json(QOPENEVENTRESULTS2 if fixn else QOPENEVENTRESULTS)  # 2018
    results2 = _load_json(QOPENEVENTRESULTS4 if fixn else QOPENEVENTRESULTS3)  # 2020
    vals = [(evid, evres['Mcat'], evres['Mw'], evres['fc']) for evid, evres in results['events'].items() if 'Mw' in evres]
    vals2 = [(evid, evres['Mcat'], evres['Mw'], evres['fc']) for evid, evres in results2['events'].items() if 'Mw' in evres]
    evid, Ml, Mw, fc = map(np.array, zip(*vals))
    evid2, Ml2, Mw2, fc2 = map(np.array, zip(*vals2))
    # load n values
    results = _load_json(QOPENEVENTRESULTS)  # 2018
    results2 = _load_json(QOPENEVENTRESULTS3)  # 2020
    nd = {evid: evres['n'] for evid, evres in results['events'].items() if 'Mw' in evres}
    nd2 = {evid: evres['n'] for evid, evres in results2['events'].items() if 'Mw' in evres}
    n = np.array([nd.get(id_) for id_ in evid], dtype=float)
    n2 = np.array([nd2.get(id_) for id_ in evid2], dtype=float)

    print(f'high frequency fall-of mean: {np.nanmean(n):.3f}  '
          f'median: {np.nanmedian(n):.3f}  std: {np.nanstd(n):.3f}')
    print(f'high frequency fall-of mean: {np.nanmean(n2):.3f}  '
          f'median: {np.nanmedian(n2):.3f}  std: {np.nanstd(n2):.3f}')
    sd = fc2stress_drop(fc, moment_magnitude(Mw, inverse=True)) / 1e6  # MPa
    sd2 = fc2stress_drop(fc2, moment_magnitude(Mw2, inverse=True)) / 1e6  # MPa

    Mwl = 0.8
    print('stress drop in MPa 2018')
    print('mean+-var sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.mean(sd), np.var(sd), 100 * np.var(sd)/np.mean(sd)))
    print('median+-mad sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.median(sd), mad(sd), 100 * mad(sd)/np.median(sd)))
    print('median+-geometric mad sd  {:.1f} -{:.1f} +{:.1f} ({:.0f}%, {:.0f}%)'.format(np.median(sd), geomad(sd)[0], geomad(sd)[1], 100 * geomad(sd)[0]/np.median(sd), 100 * geomad(sd)[1]/np.median(sd)))
    print('median sd Ml>', np.median(sd[Mw>=Mwl]))

    print('stress drop in MPa 2020')
    print('mean+-var sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.mean(sd2), np.var(sd2), 100 * np.var(sd2)/np.mean(sd2)))
    print('median+-mad sd  {:.1f}+-{:.1f} ({:.0f}%)'.format(np.median(sd2), mad(sd2), 100 * mad(sd2)/np.median(sd2)))
    print('median+-geometric mad sd  {:.1f} -{:.1f} +{:.1f} ({:.0f}%, {:.0f}%)'.format(np.median(sd2), geomad(sd2)[0], geomad(sd)[1], 100 * geomad(sd)[0]/np.median(sd), 100 * geomad(sd)[1]/np.median(sd)))
    print('median sd2', np.median(sd2), np.median(sd2[Mw2>=Mwl]))
    arr = np.asarray(list(zip(*[evid, Ml, Mw, fc, n, sd])), dtype=EPDTYPE)
    arr2 = np.asarray(list(zip(*[evid2, Ml2, Mw2, fc2, n2, sd2])), dtype=EPDTYPE)
    fname = EQ_PARAMS.format('2018' + '_n_not_fixed' * (not fixn))
    fname2 = EQ_PARAMS.format('2020' + '_n_not_fixed' * (not fixn))
    np.savetxt(fname, arr, fmt=EPFMT, header=EPHEADER)
    np.savetxt(fname2, arr2, fmt=EPFMT, header=EPHEADER)


def compare_mags():
    t1 = np.genfromtxt(EQ_PARAMS.format(2018), dtype=EPDTYPE, names=True, delimiter=',')
    t2 = np.genfromtxt(EQ_PARAMS.format(2020), dtype=EPDTYPE, names=True, delimiter=',')
    Ml = t1['Ml']
    Mw = t1['Mw']
    Ml2 = t2['Ml']
    Mw2 = t2['Mw']
    mmin, mmax = 0, np.max(Ml)
    m = np.linspace(mmin, mmax, 100)

    # method = 'least squares'
    # temp = [(r['Mcat'], r['Mw']) for id_, r in result['events'].items()
    #     if r.get('Mcat') is not None and r.get('Mw') is not None and
    #     (plot_only_ids is None or id_ in plot_only_ids)]

    # method = 'robust'

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
    ax.set_ylabel(r'moment magnitude $M_{\rm w}$')
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


def fc2stress_drop(fc, M0):
    r = 3500 * 0.21 / np.array(fc)
    stress_drop = 7 * M0 / (16 * r ** 3)
    return stress_drop


def geomad(vals):
    lmad = mad(np.log(vals))
    med = np.median(vals)
    return med-med*np.exp(-lmad), med*np.exp(lmad)-med


def _logregr(Mw, fc):
    m2, b2, _, _, m2_stderr = linregress(Mw, np.log10(fc))
    m, b = 1 / m2, -b2 / m2
    m_stderr = m2_stderr / m2 ** 2
    print(f'L2 fit, Mw independent variable: M0 ∝ fc^{1.5*m:.2f}+-{1.5*m_stderr:.2f}')
    label = f'$M_0$ ∝ {fcs}$^{{{1.5*m:.2f} \pm {1.5*m_stderr:.2f}}}$'
    return m, b, m_stderr, label



def fc_vs_Mw(fixn=True):
    dxxx = 0.26
    t1 = np.genfromtxt(EQ_PARAMS.format('2018'+ '_n_not_fixed' * (not fixn)), dtype=EPDTYPE, names=True, delimiter=',')
    t2 = np.genfromtxt(EQ_PARAMS.format('2020'+ '_n_not_fixed' * (not fixn)), dtype=EPDTYPE, names=True, delimiter=',')
    Mw1, fc1, sd1, n1 = t1['Mw'], t1['fc'], t1['stressdrop'], t1['n']
    Mw2, fc2, sd2, n2 = t2['Mw'], t2['fc'], t2['stressdrop'], t2['n']

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
    ax1.plot(fc2, Mw2, 'o', color='C1', mfc='None', ms=4, label='2020')

    m, b, _, _, m_stderr = linregress(np.log10(fc1), Mw1)
    print(f'L2 fit, fc independent variable: M0 ∝ fc^{1.5*m:.2f}+-{1.5*m_stderr:.2f}')
    m, b = _linear_fit_L1(Mw1, np.log10(fc1), m, b)
    print(f'L1 fit, fc independent variable: M0 ∝ fc^{1.5*m:.2f}')
    #m2, b2 = _linear_fit_L1(np.log10(fc1), Mw1, m, b)
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
    ax1.set_xscale('log')
    ax1.set_xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    # ax1.set_xticklabels(['', '20', '30', '40', '', '60', '', '', '', ''])
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xlim(18, 70)
    ax1.set_ylim(0.33, 1.96)
    ax1.tick_params(top=False)
    _secondary_yaxis_seismic_moment(ax1)

    # b) corner frequency histogram
    ax4 = fig.add_axes(box4, sharex=ax1)
    bins = np.logspace(np.log10(15), np.log10(70), 41)
    ax4.hist((fc2, fc1), bins=bins, histtype='barstacked', color=('C1', 'C0'), rwidth=0.9, zorder=10)
    # ax4.spines['right'].set_visible(False)
    # ax4.spines['top'].set_visible(False)
    # ax4.yaxis.set_ticks_position('left')
    # ax4.xaxis.set_ticks_position('bottom')
    ax4.tick_params(labelbottom=False)#, right=False, top=False)
    ax4.axvline(np.median(fc1), color='C0', ls='--')
    ax4.axvline(np.median(fc2), color='C1', ls='--')
    ax4.set_ylim(None, 18)

    # a) n histogram
    ax3 = fig.add_axes(box3)
    bins = np.arange(1.325, 2.31, 0.05)
    ax3.hist((n2, n1), bins=bins, histtype='barstacked', rwidth=0.9, color=('C1', 'C0'), zorder=10)
    ax3.axvline(np.nanmedian(n1), color='C0', ls='--')
    ax3.axvline(np.median(n2), color='C1', ls='--')

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
    ax5.axvline(np.median(sd2), color='C1', ls='--')
    ax5.hist((sd2, sd1), bins=bins, histtype='barstacked', color=('C1', 'C0'), rwidth=0.9, zorder=10)
    ax5.set_ylim(0, 50)

    ax5.set_xscale('log')
    ax5.set_xlabel('stress drop (MPa)')
    ax5.set_ylabel('counts')
    ax5.xaxis.set_major_formatter(ScalarFormatter())
    ax5.annotate('c)', **akwargs)

    # e) stress drop vs binned Mw
    ax6 = fig.add_axes(box6)
    Mwmid = np.arange(0.5, 1.91, 0.2)
    vals = [(sd1[np.logical_and(mwmidv-0.1 <= Mw1, Mw1 < mwmidv+0.1)],
             sd2[np.logical_and(mwmidv-0.1 <= Mw2, Mw2 < mwmidv+0.1)])
             for mwmidv in  Mwmid]
    sdv1, sdv2 = zip(*vals)
    sdm1 = np.array(list(map(np.median, sdv1)))
    sdm2 = np.array(list(map(np.median, sdv2)))
    sderr1 = np.transpose(list(map(geomad, sdv1)))
    sderr2 = np.transpose(list(map(geomad, sdv2)))
    ax6.errorbar(Mwmid-0.01, sdm1, yerr=sderr1, marker='x', ls='', label='2018')
    ax6.errorbar(Mwmid+0.01, sdm2, yerr=sderr2, marker='o', ls='', ms=4, mfc='None', label='2020')
    ratio = sdm1[:len(sdm2)] / sdm2
    print('radom stress drop ratios',  round(np.nanmedian(ratio), 2))
    ax6.set_yscale('log')
    ax6.set_ylim(0.3/1.1, 10*1.1)
    ax6.set_yticks([1, 10])
    ax6.yaxis.set_major_formatter(ScalarFormatter())

    ax6.set_ylabel('stress drop\n(MPa)')
    ax6.set_xlabel(f'moment magnitude {Mws}')
    ax6.annotate('e)', **akwargs)
    del akwargs['size']
    ax6.annotate('         median and MAD', **akwargs)
    ax6.set_xticks(Mwmid)

    @matplotlib.ticker.FuncFormatter
    def myformatter(x, pos):
        # print(x)
        return f'{x:.1f}' if 0.25 <= x <= 0.35 else ''
    ax6.yaxis.set_minor_formatter(myformatter)

    # d) fc vs binned Mw
    vals = [(fc1[np.logical_and(mwmidv-0.1 <= Mw1, Mw1 < mwmidv+0.1)],
             fc2[np.logical_and(mwmidv-0.1 <= Mw2, Mw2 < mwmidv+0.1)])
             for mwmidv in  Mwmid]
    fcv1, fcv2 = zip(*vals)
    fcm1 = np.array(list(map(np.median, fcv1)))
    fcm2 = np.array(list(map(np.median, fcv2)))
    fcerr1 = np.transpose(list(map(geomad, fcv1)))
    fcerr2 = np.transpose(list(map(geomad, fcv2)))
    ax7 = fig.add_axes(box7, sharex=ax6)
    ax7.errorbar(Mwmid-0.01, fcm1, yerr=fcerr1, marker='x', ls='')
    ax7.errorbar(Mwmid+0.01, fcm2, yerr=fcerr2, marker='o', ms=4, mfc='None', ls='')
    ratio = fcm1[:len(fcm2)] / fcm2
    print('radom corner freq ratio', round(np.nanmedian(ratio), 2))
    ax7.set_yscale('log')
    ax7.set_yticks([20, 30, 40, 50])
    ax7.yaxis.set_major_formatter(ScalarFormatter())
    ax7.tick_params(labelbottom=False)
    ax7.set_ylabel('corner frequency\n(Hz)')
    ax7.annotate('d)', **akwargs)
    ax7.annotate('         median and MAD', **akwargs)
    ax7.set_xlim(0.35, 2.05)

    # line plots for ax1 and ax7
    fclim = np.array([15, 50, 70])
    m, b, m_stderr, label = _logregr(Mw1, fc1)
    print('m, b = ', m, b)
    ax1.plot(fclim, np.log10(fclim)*m+b, zorder=-1, label=label, color='C0')

    mwlim = np.array((0.3, 2))
    m, b, m_stderr, label = _logregr(Mwmid, fcm1)
    ax7.plot(mwlim, 10**((mwlim-b)/m), '-.', zorder=-1, label=label, color='C0', alpha=0.5)
    ax1.plot(10**((mwlim-b)/m), mwlim, '-.', zorder=-1, label=label, color='C0', alpha=0.5)

    m, b, m_stderr, label = _logregr(Mw2, fc2)
    ax1.plot(fclim, np.log10(fclim)*m+b, zorder=-1, label=label, color='C1')

    m, b, m_stderr, label = _logregr(Mwmid[~np.isnan(fcm2)], fcm2[~np.isnan(fcm2)])
    ax7.plot(mwlim, 10**((mwlim-b)/m), '-.', zorder=-1, label=label, color='C1', alpha=0.5)
    ax1.plot(10**((mwlim-b)/m), mwlim, '-.', zorder=-1, label=label, color='C1', alpha=0.5)

    kw = dict(ls='--', color='0.5', zorder=-1)
    ax1.plot(fclim, fc2Mw(0.1e6, fclim), **kw)
    ax1.plot(fclim, fc2Mw(1e6, fclim), label=f'$M_0$ ∝ {fcs}$^{{-3}}$', **kw)
    ax1.plot(fclim, fc2Mw(10e6, fclim), **kw)
    ax1.legend()

    fname = '../figs/eqparams.pdf'
    if not fixn:
        fname = fname.replace('.pdf', '_n_unfixed.pdf')
    fig.savefig(fname, bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    plot_sds_wrapper()
    plot_sds_in_one()

    calc_stressdrops()
    compare_mags()
    fc_vs_Mw_vs_n()
    fc_vs_Mw()

    sm1 = moment_magnitude(0.6, inverse=True)
    sm2 = moment_magnitude(0, inverse=True)
    fac = round(sm2 ** (-1/4.7) / sm1 ** (1/-4.7), 2)
    print(f'offset of 0.6 in magnitude -> factor {fac:.2f} in corner frequency')
