# Copyright 2022 Tom Eulenfeld, MIT license

from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
from obspy import read_inventory
from qopen.core import collect_results
from qopen.imaging import plot_sites
from qopen.util import gerr

from metadata import ALL_STATIONS, BHSTATIONS


RSITES = '../qopen/02_sites/results.json'
RALLSITES = '../qopen/05_sites_all_stations/results.json'


def plot_sites_():
    inv = read_inventory(ALL_STATIONS)
    with open(RSITES) as f:
        results = json.load(f)
    def sortkey(sta):
        return ('A%02d' % BHSTATIONS.index(sta.split('.')[-1])
                if sta.split('.')[-1] in BHSTATIONS else sta)
    plot_sites(results, nx=9, figsize=(12, 4.5), ylim=(10**-2.2, 10**2.2),
               annotate=False, ylabel=None, show_excluded=False, sortkey=sortkey)  # excluded: HE.PVF
    fig = plt.gcf()
    fig.axes[0].set_yticks((0.01, 0.1, 1, 10, 100))
    results['R'].pop('HE.PVF')
    assert len(results['R']) == len(fig.axes[:-2])
    for ax, sta in zip(fig.axes[:-2], sorted(results['R'], key=sortkey)):
        if sta in ('HE.MALM', 'HE.RUSK'):
            ax.axhline(0.25, lw=1, color='C1', zorder=50)
        elif sta.split('.')[1] in BHSTATIONS:
            ax.axhline(0.25, lw=1, color='0.5', zorder=50)
        else:
            ax.axhline(1, lw=1, color='0.5', zorder=50)
        if sta.split('.')[1] in BHSTATIONS:
            depth = inv.get_coordinates(sta + '..HHZ')['local_depth']
            label = f'{sta} {depth:.0f}m'
        else:
            label = sta
        ax.annotate(label, (0, 1), (3, -3), 'axes fraction',
                    'offset points', ha='left', va='top', size='x-small')
    fig.supylabel('energy site amplification R', x=0.07)
    fig.savefig('../figs/sites2.pdf', bbox_inches='tight', pad_inches=0.1)


def load_data():
    with open(RALLSITES) as f:
        results2 = json.load(f)
    all_results = deepcopy(results2)
    for evid, ev in results2['events'].items():
        ev['R'] = {sta: r2 for sta, r2 in ev['R'].items() if sta.startswith('OT.E') or sta in ['HE.ELFV']}
    return results2, all_results


def plot_map(res, fname, evarray=False):
    inv = read_inventory(ALL_STATIONS)
    fs = res['freq']
    R = collect_results(res, only=['R'])['R']
    coords = []
    ameans = []
    for station in sorted(R):
        if np.all(np.isnan(R[station])):
            continue
        seedid = station + ('..HHZ' if station.startswith('HE') else '..DPZ')
        print(seedid)
        coord = inv.get_coordinates(seedid)
        if evarray and station == 'HE.ELFV':
            continue
            coord['latitude'] = 60.203
            coord['longitude'] = 24.8195
        coords.append((coord['longitude'], coord['latitude']))
        means, err1, err2 = gerr(R[station], axis=0, robust=True)
        print(station)
        print(coord)
        print(means)
        ameans.append(means)
    print(len(fs))
    fig = plt.figure(figsize=(6, 10))
    ameans = np.array(ameans, dtype=float)
    j = 1
    for i, f in list(enumerate(fs))[:-1:2]:
        ax = fig.add_subplot(3, 2, j)
        j += 1
        im = ax.scatter(*zip(*coords), s=None, c=np.log10(ameans[:, i]), marker='v',
                        cmap=None if evarray else 'turbo')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.annotate('{:.1f} Hz'.format(f), (0.9, 0.1), xycoords='axes fraction', ha='right')
        if not evarray:
            ax.set_xlim(None, 25.13)
            ax.set_ylim(None, 60.26)
    cbar = fig.colorbar(im, cax=fig.add_axes((0.1, 0.03, 0.8, 0.02)), orientation='horizontal')
    cbar.set_ticks([])
    cbar.ax.annotate('low', (0, -0.5), xycoords='axes fraction', va='top')
    cbar.ax.annotate('high', (1, -0.5), xycoords='axes fraction', va='top', ha='right')
    fig.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    plot_sites_()
    evres, allres = load_data()
    res = allres
    plot_map(evres, '../figs/sites_ev_map.pdf', evarray=True)
    plot_map(allres, '../figs/sites_all_map.pdf')
    plot_sites(evres, fname='../figs/sites_ev.pdf', figsize=(10, 16), mean='robust')
    plot_sites(allres, fname='../figs/sites_all.pdf', figsize=(10, 16), mean='robust')
