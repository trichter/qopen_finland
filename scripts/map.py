# Copyright 2022 Tom Eulenfeld, MIT license

import cartopy.crs as ccrs
from cartopy.io.img_tiles import Stamen
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from obspy import read_inventory
try:
    from obspy.io.csv import load_csv
except ImportError:
    from obspycsv import load_csv
import shapely.geometry as sgeom

from metadata import BHSTATIONS, STATIONS, EVENTS2018, EVENTS2020
from util import add_scale

# https://epsg.io/3035
EA_EURO = ccrs.LambertAzimuthalEqualArea(
    central_longitude=10, central_latitude=52,
    false_easting=4321000, false_northing=3210000,
    globe=ccrs.Globe(ellipse='GRS80'))

GEO = ccrs.Geodetic()

ST1_TOWER_OF_DOOM = [24.827126, 60.188370]
X0, Y0  = 25490448, 6675071

# PVF is out of map bounds, it is excluded from analysis because it is too far away for the used time windows
DONOTPLOT = 'PVF'
MAPINSET = 'NUR MEF MALM HEL5 DT00 DT01'.split()
BOTTOM = 'RAD DT00 ELFV HEL5 HEL2'.split()


def inside(lat, lon, extent):
    return extent[2] <= lat <= extent[3] and extent[0] <= lon <= extent[1]

def load_wellpath(fname):
    data = np.genfromtxt(fname, skip_header=1)
    return data[:, 0]-X0, data[:, 1]-Y0, data[:, 2]

def _mag2size(mag):
    return 5 + 3*(mag + 1.4) ** 2

def latlon2xy(data):
    from pyproj import Proj, Transformer
    trans =  Transformer.from_proj(Proj('epsg:4326', proj='latlong'), 'epsg:3879', always_xy=True)
    x, y = trans.transform(data['lon'], data['lat'])
    x = x - X0
    y = y - Y0
    print('median x, y, z, {:.2f}, {:.2f}, {:.4f}, max mag {}'.format(
        np.median(x), np.median(y), np.median(np.abs(data['dep'])), max(data['mag'])))
    return {'xs': x, 'ys': y, 'zs': 1000*data['dep'], 's': _mag2size(data['mag'])}


def plot_eqcloud(ax=None):
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect(aspect = (1,1,2))
    ax.plot(*load_wellpath('../data/wellpath2018.txt'))
    ax.plot(*load_wellpath('../data/wellpath2020.txt'))
    ev1 = load_csv('../data/events2018.csv')
    ev2 = load_csv('../data/events2020.csv')
    ax.scatter(**latlon2xy(ev1), alpha=0.5)
    ax.scatter(**latlon2xy(ev2), alpha=0.5)
    ax.invert_zaxis()
    ax.set_xlabel('easting (m)', labelpad=-2)
    ax.set_ylabel('northing (m)')
    ax.set_zlabel('depth (m)', labelpad=25)
    ax.tick_params(axis='x', which='major', pad=-2)
    ax.tick_params(axis='z', which='major', pad=12)
    ax.set_xlim(-400, 1000)
    ax.set_ylim(-200, 1200)
    ax.set_zlim(6500, 4500)
    # ax.view_init(elev=10, azim=-35)
    # ax.view_init(elev=20, azim=-100)
    ax.view_init(elev=15, azim=-75)
    ax.xaxis.set_major_locator(MultipleLocator(400))
    ax.yaxis.set_major_locator(MultipleLocator(400))
    ax.zaxis.set_major_locator(MultipleLocator(400))

    ax.annotate('d)', (0.2, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='x-large')
    from matplotlib.lines import Line2D
    Mls = r'$M_{\rm{L}}$'
    legend_elements = [
        Line2D([0], [0],marker='o', linestyle='', mfc='none', color='k', label=f'{Mls}={mag:.1f}', markersize=_mag2size(mag)**0.5)
        for mag in [-1, 0, 1]]
    ax.legend(bbox_to_anchor=(0.41, 0.25), handles=legend_elements, fontsize='small', frameon=False, handletextpad=0.4)
    fig.savefig('../figs/eqcloud.pdf', bbox_inches='tight')
    import os
    os.system("pdfcrop --margins '5 5 5 5' ../figs/eqcloud.pdf ../figs/eqcloud.pdf")


def plot_map():
    fig = plt.figure(figsize=(10, 10))
    tiler = Stamen('terrain')
    # tiler2 = Stamen('terrain-background')
    crs = tiler.crs
    ax = fig.add_subplot(111, projection=crs) #EA_EURO
    extent = [24.7, 25.1, 60.125, 60.265]  # map in fig 1a
    extent2 = [24, 25.6, 59.85, 60.65]  # map in fig 1b
    extent3 = [15, 35, 57, 72]  # map in fig 1c
    ax.set_extent(extent, crs=GEO)
    ax.add_image(tiler, 12, interpolation='spline36', alpha=0.6)

    fig.canvas.draw()
    box = ax.get_position().bounds
    subax = fig.add_axes([box[0]+box[2]-0.24, box[1]+box[3]-0.21, 0.24, 0.19], projection=crs)
    subax3 = fig.add_axes([box[0]+box[2]-0.22, box[1]+0.02, 0.2, 0.28], projection=crs)
    subax.set_extent(extent2, GEO)
    subax3.set_extent(extent3, GEO)
    subax.add_geometries([sgeom.box(extent[0], extent[2], extent[1], extent[3])], GEO,
                          facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)
    subax3.add_geometries([sgeom.box(extent2[0], extent2[2], extent2[1], extent2[3])], GEO,
                          facecolor='none', edgecolor='k', linewidth=1, alpha=0.5)

    subax.add_image(tiler, 10, interpolation='spline36', alpha=0.7)
    subax3.add_image(tiler, 5, interpolation='spline36')

    ax.scatter(*ST1_TOWER_OF_DOOM, s=25, marker='o', edgecolors='k', facecolors='none', label='well head', zorder=11, transform=GEO)
    inv = read_inventory(STATIONS)
    stas = [(station.code, station.latitude, station.longitude)
            for net in inv for station in net if station.code != DONOTPLOT]
    OT = [station.code for net in inv for station in net if net.code == 'OT']
    ev1 = load_csv(EVENTS2018)
    ev2 = load_csv(EVENTS2020)
    ax.plot(ev1['lon'], ev1['lat'], 'C0o', ms=2, transform=GEO, label='2018 earthquakes')
    ax.plot(ev2['lon'], ev2['lat'], 'C1o', ms=2, transform=GEO, label='2020 earthquakes')

    ids, slat, slon = zip(*stas)
    kw = dict(marker='^', color='none', transform=GEO, s=100, lw=2)
    for cond, label, color, zorder in [
            (lambda x: x in BHSTATIONS, 'St1 borehole stations', 'C3', 10),
            (lambda x: x not in OT + BHSTATIONS, 'HEL surface stations', 'C1', 8),
            (lambda x: x in OT, 'OT short-period stations', 'C4', 6)]:

        slon2, slat2 = zip(*[c[1:] for c in zip(ids, slon, slat) if cond(c[0]) and c[0] not in MAPINSET])
        ax.scatter(slon2, slat2, edgecolors=color, label=label, zorder=zorder, **kw)
        slon2, slat2 = zip(*[c[1:] for c in zip(ids, slon, slat) if cond(c[0]) and c[0] in MAPINSET])
        subax.scatter(slon2, slat2, edgecolors=color, zorder=zorder, **kw)

    for id1, lat1, lon1 in stas:
        axp = subax if id1 in MAPINSET else ax
        # xy = (2, 2) if s.station not in ('KOPD', 'KAC') else (-10, 5)
        pos = (-2, -6) if id1 == 'HEL2' else (2, 2) if id1 not in BOTTOM else (2, -6)
        va = 'top' if id1 in BOTTOM else 'bottom'
        ha = 'right' if id1 == 'HEL2' else 'left'
        axp.annotate(id1, (lon1, lat1), pos, GEO._as_mpl_transform(axp), 'offset points', size='x-small', zorder=10, va=va, ha=ha)

    add_scale(ax, 5, (24.75, 60.13))
    add_scale(subax, 40, (25.2, 59.9))

    leg = ax.legend(loc=(0.26, 0.02), ncol=2, framealpha=0.35, fontsize='small',
                    handleheight=1.1, borderpad=0.6, numpoints=3)
    xd = [6., 8.5, 10.5]
    yd = [3.5, 5.8, 4]
    for l in leg.legendHandles[1:3]:
        l.set_data(xd, yd)
    for ax_, label in zip([ax, subax, subax3], 'abc'):
        ax_.annotate(label + ')', (0, 1), (8, -6), 'axes fraction', 'offset points', va='top', size='x-large')
    fig.savefig('../figs/map.pdf', bbox_inches='tight', pad_inches=0.1, dpi=100)
    plt.show()


if __name__ == '__main__':
    plot_map()
    plot_eqcloud()
