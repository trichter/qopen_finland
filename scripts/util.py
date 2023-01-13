# Copyright 2022 Tom Eulenfeld, MIT license

import cartopy.crs as ccrs
import collections
from itertools import cycle
from matplotlib.legend_handler import HandlerBase, _Line2DHandleList
import numpy as np


GEO = ccrs.Geodetic()


def _eventid(event):
    return str(event.resource_id).split('/')[-1]


def add_scale(ax, length, loc, crs=GEO, lw=1,
              cap=2, label=True, size=None, vpad=2):
    bx, by = ax.projection.transform_point(loc[0], loc[1], crs)
    bx1, bx2 = bx - 500 * length, bx + 500 * length
    ax.plot((bx1, bx2), (by, by), color='k', linewidth=lw)
    if cap:
        kw = {'xycoords': 'data', 'textcoords': 'offset points', 'arrowprops':
              {'arrowstyle': '-', 'connectionstyle': 'arc3',
               'shrinkA': 0, 'shrinkB': 0, 'linewidth': lw}}
        ax.annotate('', (bx1, by), (0, cap), **kw)
        ax.annotate('', (bx1, by), (0, -cap), **kw)
        ax.annotate('', (bx2, by), (0, cap), **kw)
        ax.annotate('', (bx2, by), (0, -cap), **kw)
    if label:
        ax.annotate(str(length) + ' km', (bx, by), (0, vpad), size=size,
                    textcoords='offset points', ha='center', va='bottom')


class MyHandlerTuple(HandlerBase):
    """
    Handler for Tuple with possiblity to plot above each other (use mdivide)

    Copied and modified from matplotlib/legend_handler.py
    """

    def __init__(self, ndivide=1, mdivide=1, pad=None, **kwargs):
        """
        Parameters
        ----------
        ndivide : int, default: 1
            The number of sections to divide the legend area into.  If None,
            use the length of the input tuple.
        pad : float, default: :rc:`legend.borderpad`
            Padding in units of fraction of font size.
        **kwargs
            Keyword arguments forwarded to `.HandlerBase`.
        """
        self._ndivide = ndivide
        self._mdivide = mdivide
        self._pad = pad
        super().__init__(**kwargs)

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        handler_map = legend.get_legend_handler_map()

        if self._ndivide is None:
            ndivide = len(orig_handle)
        else:
            ndivide = self._ndivide
        if self._mdivide is None:
            mdivide = len(orig_handle)
        else:
            mdivide = self._mdivide

        if self._pad is None:
            pad = legend.borderpad * fontsize
        else:
            pad = self._pad * fontsize

        if ndivide > 1:
            width = (width - pad * (ndivide - 1)) / ndivide
        if mdivide > 1:
            height = (height - pad * (ndivide - 1)) / ndivide

        xds_cycle = cycle(xdescent - (width + pad) * np.arange(ndivide))
        yds_cycle = cycle(ydescent - (height + pad) * np.arange(mdivide) + pad)

        a_list = []
        for handle1 in orig_handle:
            handler = legend.get_legend_handler(handler_map, handle1)
            _a_list = handler.create_artists(
                legend, handle1,
                next(xds_cycle), next(yds_cycle),
                width, height, fontsize, trans)
            if isinstance(_a_list, _Line2DHandleList):
                _a_list = [_a_list[0]]
            a_list.extend(_a_list)

        return a_list


class IterMultipleComponents(object):

    """
    Return iterable to iterate over associated components of a stream.

    :param stream: Stream with different, possibly many traces. It is
        split into substreams with the same seed id (only last character
        i.e. component may vary)
    :type key: str or None
    :param key: Additionally, the stream is grouped by the values of
         the given stats entry to differentiate between e.g. different events
         (for example key='starttime', key='onset')
    :type number_components: int, tuple of ints or None
    :param number_components: Only iterate through substreams with
         matching number of components.
    """

    def __init__(self, stream, key=None, number_components=None):
        substreams = collections.defaultdict(stream.__class__)
        for tr in stream:
            k = (tr.id[:-1], str(tr.stats[key]) if key is not None else None)
            substreams[k].append(tr)
        n = number_components
        self.substreams = [s for _, s in substreams.items()
                           if n is None or len(s) == n or
                           (not isinstance(n, int) and len(s) in n)]

    def __len__(self):
        return len(self.substreams)

    def __iter__(self):
        for s in self.substreams:
            yield s
