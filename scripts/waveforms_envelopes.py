# Copyright 2021 Tom Eulenfeld, MIT license

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pickle
from obspy import read, read_events, read_inventory
from qopen.core import get_pair, Gsmooth, observed_energy, filter_width
from qopen.util import smooth
from qopen.rt import G as G_func

from metadata import BHSTATIONS, EVENTS2018, STATIONS
from util import _eventid, IterMultipleComponents


FIGLABELKW = dict(xy=(0, 1), xytext=(5, -5),
                  xycoords='axes fraction', textcoords='offset points',
                  va='top', size='x-large')
OTIME = 60
EVID = '2018171232614IMS000000'
OUT = '../figs'
OUT2 = '../figs/more_envelopes'


def set_gridlabels(ax, i, n, N, xlabel='frequency (Hz)', ylabel=None, y=None):
    if i % n != 0 and ylabel:
        plt.setp(ax.get_yticklabels(), visible=False)
    # elif i // n == (n - 1) // 2 and ylabel:
    #     ax.set_ylabel(ylabel)
    elif i == 7:
        ax.set_ylabel(ylabel, y=y)
    if i < N - n and xlabel:
        plt.setp(ax.get_xticklabels(), visible=False)
    elif i % n == (n - 1) // 2 and i >= N - n - 1 and xlabel:
        ax.set_xlabel(xlabel)


def _get_times(tr):
    t0 = tr.stats.starttime - tr.stats.origintime
    return np.arange(len(tr)) * tr.stats.delta + t0


def plot_fits(energies, g0, b, W, R, v0, info, smooth=None,
              smooth_window='bartlett',
              ylim=None, tws=None, yticks=None, y=None, ylabel=None,
              bbox='tight'):
    from metadata import BHSTATIONS
    def sortkey(sta):
        return 'A%02d' % BHSTATIONS.index(sta.split('.')[-1]) if sta.split('.')[-1] in BHSTATIONS else sta
    def sortkey2(item):
        i, energy = item
        evid, station = get_pair(energy)
        return sortkey(station)

    fs = 250 / 25.4
    plt.figure(figsize=(fs, 0.5*fs))
    tcoda, tbulk, Ecoda, Ebulk, Gcoda, Gbulk = info
    N = len(energies)
    nx, ny = 7, 5
    gs = gridspec.GridSpec(ny, nx, wspace=0.06, hspace=0.08)
    share = None
    if b is None:
        b = 0
    c1 = 'mediumblue'
    c2 = 'darkred'
    c1l = '#8181CD'
    c2l = '#8B6969'
    for j, (i, energy) in enumerate(sorted(enumerate(energies), key=sortkey2)):
        evid, station = get_pair(energy)
        ax = plt.subplot(gs[j // nx, j % nx], sharex=share, sharey=share)
        plot = ax.semilogy

        def get_Emod(G, t):
            return R[station] * W[evid] * G * np.exp(-b * t)
        st = energy.stats
        r = st.distance
        t = _get_times(energy) + r / v0 - (st.sonset - st.origintime)

        if smooth:
            plot(t, energy.data_unsmoothed, color='0.7')
        plot(t, energy.data, color=c1l)
        G_ = Gsmooth(G_func, r, t, v0, g0, smooth=smooth,
                     smooth_window=smooth_window)
        Emod = get_Emod(G_, t)
        index = np.argwhere(Emod < 1e-30)[-1]
        Emod[index] = 1e-30

        plot(t, Emod, color=c2l)

        plot(tcoda[i], Ecoda[i], color=c1)
        Emodcoda = get_Emod(Gcoda[i], tcoda[i])
        plot(tcoda[i], Emodcoda, color=c2)

        if tbulk and len(tbulk) > 0:
            plot(tbulk[i], Ebulk[i], 'o', color=c1, mec=c1, ms=4)
            Emodbulk = get_Emod(Gbulk[i], tbulk[i])
            plot(tbulk[i], Emodbulk, 'o', ms=3,
                 color=c2, mec=c2)

        l = '%s\n%dkm' % (station, r / 1000)
        ax.annotate(l, (1, 1), (-5, -5), 'axes fraction',
                    'offset points', ha='right', va='top', size='x-small')

        if ylabel is None:
            ylabel = 'spectral energy density $E$ (Jm$^{-3}$Hz$^{-1}$)                               '
        set_gridlabels(ax, j, nx, N, xlabel='time (s)', ylabel=ylabel, y=y)
        if tws:
            kw = dict(color='darkgreen', alpha=0.5, lw=0, zorder=10000)
            tw = tws[0]
            if tw is not None:  # direct S wave window
                ax.axvspan(tcoda[i][0]+tw[0], tcoda[i][0]+tw[1], 0.05, 0.08, **kw)
            tw = tws[1]
            if tw is not None:  # coda window
                ax.axvspan(tcoda[i][0]+tw[0], tcoda[i][-1]+tw[1], 0.05, 0.08, **kw)
        if share is None:
            share = ax
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim((-2, 28))
    ax.set_xticks([0, 10, 20])
    if yticks:
        ax.set_yticks(10. ** np.array(yticks))
    if ylim:
        ax.set_ylim(ylim)


def plot_fits_wrapper(out=OUT):
    from matplotlib.transforms import Bbox
    evid = f'{EVID}_16.00Hz-32.00Hz'
    fname = f'../qopen/01_go/fits_{evid}.pkl'
    with open(fname, 'rb') as f:
        tup = pickle.load(f)
    plot_fits(*tup,
              ylim=(1e-13 / 1.5, 1e-6 * 1.5),
              yticks=(-12, -10, -8),
              y=0.5,
              tws=[(-4, -0.3), (0.3, 0)]
              )
    bbox = Bbox([[0.4, -0.05], [8.95, 4.45]])
    plt.savefig(f'{out}/qopen_fits_{evid}.pdf', bbox_inches=bbox)

    ylabel = 'spectral energy density $E$ (Jm$^{-3}$Hz$^{-1}$)                     '
    ylabel = '$E$ (Jm$^{-3}$Hz$^{-1}$)'
    evid = '2020105023127ISUHX00000_16.00Hz-32.00Hz'
    fname = f'../qopen/06_source_2020/fits_{evid}.pkl'
    with open(fname, 'rb') as f:
        tup = pickle.load(f)
    plot_fits(*tup,
              ylim=(1e-15 / 1.5, 1e-8 * 1.5),
              yticks=(-13, -11, -9),
              y=1,
              tws=[None, (0, 0)],
              ylabel=ylabel
              )
    bbox = Bbox([[0.4, 2.3], [8.95, 4.45]])
    plt.savefig(f'{out}/qopen_fits_{evid}.pdf', bbox_inches=bbox)


def _load_data_remove_sensitivity(fname, check_len=False):
    stream = read(fname)
    stream.detrend('linear')
    inv = read_inventory(STATIONS)
    stas = {sta.code for net in inv for sta in net}
    stream.traces = [tr for tr in stream if tr.stats.station in stas]
    traces = []
    for tr in stream:
        try:
             tr.remove_sensitivity(inv)
        except:
            print('remove response failed for', tr.id)
        else:
            traces.append(tr)
    stream.traces = traces
    if check_len and len(stream) != 3:
        msg = f'stream in file {fname} has {len(stream)}!=3 traces'
        print(msg)
        raise ValueError(msg)
    return stream


def _stream2evelope(stream, filter_):
    sr = stream[0].stats.sampling_rate
    df = filter_width(sr, **filter_)
    energy = observed_energy(stream, 2700, df=df)
    energy.data = smooth(energy.data, int(round(sr * 1)), window='flat', method='zeros')
    energy.decimate(int(sr) // 10, no_filter=True)
    return energy


def plot_waveforms(evid=EVID, out=OUT):
    fname = f'../data/2018_IMS/EVENT_DATA/{evid}/{evid}.mseed'
    stream = _load_data_remove_sensitivity(fname)

    # helper to get distance sorting order based on trigger
    # from obspy.signal.trigger import ar_pick
    # def sort_tr(sta):
    #     tr2 = stream.select(component='Z', station=sta)[0]
    #     tr3 = stream.select(component='N', station=sta)[0]
    #     tr4 = stream.select(component='E', station=sta)[0]
    #     df = tr2.stats.sampling_rate
    #     p_pick, _ = ar_pick(tr2.data, tr3.data, tr4.data, df,
    #                         10.0, 50.0, 1.0, 0.1, 4.0, 1.0, 2, 8, 0.1, 0.2, False)
    #     return p_pick
    # stream.traces = sorted(stream.traces, key=lambda tr: sort_tr(tr.stats.station))
    # print(' '.join({tr.stats.station: None for tr in stream}.keys()))

    sorttr2 = ('MURA OTRA TAGC ELFV HEL2 DID RAD EV00 HEL3 ZAK LEPP LTS SS01 WEG '
               'HAN HEL1 PM00 TAPI MKK RUSK TVJP PK00 TL01 LASS KUN RS00 HEL4 MUNK '
               'UNIV MALM HEL5 DT00 DT01 MEF NUR PVF').split()
    stream.traces = sorted(stream.traces, key=lambda tr: sorttr2.index(tr.stats.station))
    filter_ = dict(corners=2, zerophase=True, freq=1, type='highpass')
    stream.filter(**filter_)
    fig = plt.figure(figsize=(7, 9))
    ax = fig.add_subplot(111)
    stream.normalize()
    dy = 0.5*np.median(np.abs(stream.max()))
    for i, stream3c in enumerate(IterMultipleComponents(stream)):
        for j, tr in enumerate(stream3c):
            if tr.stats.station == 'MUNK' and tr.stats.component == 'E':
                tr.data = 0.2 * tr.data
            offset = dy * (4*i+j)
            ax.plot(tr.times()-OTIME, tr.data + offset, lw=0.5, color='0.5', zorder=-10)
        sta = tr.id.rsplit('.', 2)[0]
        ax.annotate(sta, (14, offset), ha='right', va='top', fontsize='x-small',
                    bbox=dict(boxstyle="round", fc='w', ec='None', alpha=0.5))
    ax.set_rasterization_zorder(-9)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('normalized ground velocity')
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-1, offset+1)
    ax.set_yticks([])
    fig.savefig(f'{out}/waveforms_{evid}.pdf', bbox_inches='tight', dpi=150)


def plot_envelopes(evid=EVID, out=OUT, label=False):
    fname = f'../data/2018_IMS/EVENT_DATA/{evid}/{evid}.mseed'
    #fname = f'../data/waveforms2018/{evid}_*.mseed'
    stream = _load_data_remove_sensitivity(fname)
    fmin, fmax = 8, 16
    filter_ = dict(corners=2, zerophase=True, freqmin=fmin, freqmax=fmax, type='bandpass')
    stream.filter(**filter_)
    trcs = []
    for stream3c in IterMultipleComponents(stream, key='starttime', number_components=3):
        energy = _stream2evelope(stream3c, filter_)
        trcs.append(energy)
    stream.traces = trcs

    fig = plt.figure()
    ax = fig.add_subplot(111)
    arrowprops = dict(arrowstyle='->', lw=2)
    for tr in stream:
        ax.semilogy(tr.times()-OTIME, tr.data, lw=0.5, color='0.35')
    ax.annotate('diffuse Moho\nreflection', (20,  4e-11), (20, 1e-8),
                ha='center', arrowprops=arrowprops)
    ax.annotate('    Conrad', (10,  4e-11), (10, 2e-9),
                ha='center', arrowprops=arrowprops)
    ax.set_xlabel('time (s)')
    ylabel = 'spectral energy density $E$ (Jm$^{-3}$Hz$^{-1}$)'
    ax.set_ylabel(ylabel)
    ax.set_xlim(-25, 95)
    ax.annotate('a)', **FIGLABELKW)
    fig.savefig(f'{out}/envelopes_{evid}_{fmin:.2f}Hz-{fmax:.2f}Hz.pdf', bbox_inches='tight')


def plot_envelopes2(evid=EVID, stas=('MALM',), ext='pdf', out=OUT,
                    normalize=False):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.colorbar import ColorbarBase

    fname = f'../data/2018_IMS/EVENT_DATA/{evid}/{evid}.mseed'
    try:
        stream = _load_data_remove_sensitivity(fname)
    except:
        return
    for sta in stas:
        stream2 = stream.select(station=sta)
        if len(stream2) == 0:
            print(f'No data for station {sta}')
            continue
        assert len(stream2) == 3
        fcs = np.array([3.0, 4.2, 6.0, 8.5, 12.0, 17.0, 24.0, 33.9, 48.0, 67.9, 96.0, 135.8, 192.0])
        fc_bounds = np.append(fcs * (5/6), fcs[-1:]*(7/6))

        # color map
        cmap = plt.get_cmap('plasma_r')
        cmapv = np.linspace(0.2, 1, len(fcs))
        colors = cmap(cmapv)
        cmap = ListedColormap(colors)
        norm = BoundaryNorm(fc_bounds, ncolors=len(fcs))
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax7 = fig.add_axes([0.56, 0.82, 0.3, 0.01])
        cbar = ColorbarBase(ax7, cmap=cmap, norm=norm, orientation='horizontal')#, extend='max', spacing='proportional')#, extendfrac=0.2)
        cbar.set_ticks(fcs[::2])
        cbar.set_label('central frequency (Hz)')

        trcs = []
        for fc in fcs:
            fmin = fc * (1 - 1/3)
            fmax = fc * (1 + 1/3)
            filter_ = dict(corners=2, zerophase=True, freqmin=fmin, freqmax=fmax, type='bandpass')
            if fc == 192.0:
                filter_ = dict(corners=2, zerophase=True, freq=fmin, type='highpass')
            stream3 = stream2.copy().filter(**filter_)
            assert len(stream3) == 3
            energy = _stream2evelope(stream3, filter_)
            sta = energy.stats.station
            if sta not in stas:
                continue
            trcs.append(energy)
        stream2.traces = trcs
        if normalize:
            stream2.normalize()
        for c, tr in zip(colors, stream2):
            ax.semilogy(tr.times()-OTIME, tr.data, lw=0.5, color=c)#, color='0.35')
        ax.set_xlabel('time (s)')
        ylabel = 'spectral energy density $E$ (Jm$^{-3}$Hz$^{-1}$)'
        ax.set_ylabel(ylabel)
        ax.set_xlim(-25, 95)
        ax.annotate('b)', **FIGLABELKW)
        fig.savefig(f'{out}/envelopes_{evid}_{sta}{"_normalized" if normalize else ""}.{ext}', bbox_inches='tight')


def plot_envelopes3(evid=EVID, out=OUT):
    trcs = []
    fname = f'../data/2018_IMS/EVENT_DATA/{evid}/{evid}.mseed'
    try:
        stream = _load_data_remove_sensitivity(fname)
    except:
        return
    for sta in BHSTATIONS:
        stream2 = stream.select(station=sta)
        if len(stream2) == 0:
            print(f'No data for station {sta}')
            continue
        assert len(stream2) == 3
        fc = 192*(1-1/3)
        filter_ = dict(corners=2, zerophase=True, freq=fc, type='highpass')
        stream3 = stream2.copy().filter(**filter_)
        energy = _stream2evelope(stream3, filter_)
        trcs.append(energy)
    stream.traces = trcs
    fig = plt.figure()
    ax = fig.add_subplot(111)
    arrowprops = dict(arrowstyle='->', lw=2)
    for tr in stream:
        ax.semilogy(tr.times()-OTIME, tr.data, lw=0.5, color='0.35')
    ax.annotate('diffuse Moho\nreflection', (20,  2e-11), (20, 1e-8),
                ha='center', arrowprops=arrowprops)
    ax.annotate('', (10,  2e-11), (10, 2e-10),
                arrowprops=arrowprops)
    ax.set_xlabel('time (s)')
    ylabel = 'spectral energy density $E$ (Jm$^{-3}$Hz$^{-1}$)'
    ax.set_ylabel(ylabel)
    ax.set_xlim(-25, 95)
    fig.savefig(f'{out}/envelopes_{evid}_>128Hz.pdf')
    plt.close()


if __name__ == '__main__':
    plot_fits_wrapper()
    plot_waveforms()

    plot_envelopes()
    plot_envelopes2()
    plot_envelopes2(normalize=True, out=OUT2)
    plot_envelopes3(out=OUT2)
    plot_envelopes2(stas=BHSTATIONS, out=OUT2, ext='png')
    plt.close('all')

    # load event ids of ML>=1 2018 events
    events = read_events(EVENTS2018)
    events = events.filter('magnitude >= 1')
    evids = [_eventid(event) for event in events]
    for evid in evids:
        print('evid', evid)
        plot_envelopes2(evid=evid, out=OUT2, ext='png')
        plt.close()
