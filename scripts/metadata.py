# Copyright 2022 Tom Eulenfeld, MIT license

from obspy import read_events, read_inventory
from util import _eventid

EVENTS2018 = '../data/events2018_ML>=0.csv'
EVENTS2018_WPICKS = '../data/events2018_ML>=0.csz'
EVENTS2020 = '../data/events2020_ML>=0.csv'
ALL_STATIONS = '../data/stations2018.xml'
STATIONS = '../data/selected_stations.xml'

BHSTATIONS = 'MALM RUSK ELFV LASS LEPP MUNK MURA OTRA TAGC TAPI TVJP UNIV'.split()


def select_inv():
    inv = read_inventory(ALL_STATIONS)
    network_codes = {}
    for net in inv:
        for sta in net:
            network_codes[sta.code] = net.code
    print('all stations')
    print(inv)
    stations = ['MEF',
                'PVF',
                'NUR',
                'HEL?',
                'DID', 'DT0?', 'HAN', 'KUN', 'LTS', 'MKK', 'RAD', 'WEG', 'ZAK',
                'PK00', 'PM00', 'RS00',  # 00-03
                'EV00', 'SS01', 'TL01',  # 00-23, 00-24, 01-22
                ] + BHSTATIONS
    inv = sum((inv.select(station=sta) for sta in stations), inv.select(station='XXX'))
    print('\nselected stations')
    print(inv)
    inv.write(STATIONS, 'STATIONXML')
    inv.write(STATIONS.replace('.xml', '.txt'), 'STATIONTXT')
    return network_codes


def plot_response():
    inv = read_inventory(STATIONS)
    inv = (inv.select(station='SS01', channel='*Z') +
           inv.select(station='HEL1', channel='*Z') +
           inv.select(station='MEF', channel='*Z') +
           inv.select(station='NUR', channel='*Z') +
           inv.select(station='ELFV', channel='*Z')
           )
    inv.plot_response(0.1, outfile='../figs/response.pdf')



def prepare_events(network_codes=None):
    kw = dict(names='id year mon day hour minu sec _ lat lon dep _ _ mag',
              default = {'magtype': 'Ml'},
              skipheader=1,
              format='CSV')
    # first prepare 2020 catalog
    events = read_events('../data/2020_ISUH_fully_picked/EVENT_METADATA/CSV/2020_ISUH_events.csv', **kw)
    # filter out one natural earthquake
    events.events = [ev for ev in events if _eventid(ev) != '2020204003527ISUHX00000']
    events.write('../data/events2020.csv', 'CSV')
    events = events.filter('magnitude >= 0')
    events.write('../data/events2020_ML>=0.csv', 'CSV')

    # prepare 2018 catalog
    # use the IMS catalog with precise picks and origins
    events = read_events('../data/2018_IMS/EVENT_METADATA/CSV/IMS_2018_events.csv', **kw)
    events.write('../data/events2018.csv', 'CSV')
    # discard the magnitudes and add magnitudes from the ISUH 2018 catalog
    # -> magnitudes are estimated with the same procedure in 2018 and 2020
    print('add magnitudes from ISUH events')
    events2 = read_events('../data/2018_ISUH_fully_picked/EVENT_METADATA/CSV/2018_ISUH_fully_picked_events.csv', **kw)
    mags2 = {_eventid(ev).replace('ISUHX', 'IMS0'): ev.magnitudes[0].mag for ev in events2}
    events3 = []
    for ev in events:
        evid = _eventid(ev)
        if evid in mags2:
            ev.magnitudes[0].mag = mags2[evid]
            events3.append(ev)
    events.events = events3
    events = events.filter('magnitude >= 0')
    events.write('../data/events2018_ML>=0.csv', 'CSV')

    print('add picks from QuakeML files...')
    for ev in events:
        evid = _eventid(ev)
        fname = f'../data/2018_IMS/EVENT_DATA/{evid}/{evid}.xml'
        pev = read_events(fname)[0]
        ev.origins[0].arrivals = pev.origins[0].arrivals
        ev.picks = pev.picks
        # correct wrong network codes in picks
        if network_codes is not None:
            for pick in ev.picks:
                pick.waveform_id.network_code = network_codes[pick.waveform_id.station_code]
    events.write('../data/events2018_ML>=0.csz', 'CSZ')


if __name__ == '__main__':
    network_codes = select_inv()
    plot_response()
    prepare_events(network_codes=network_codes)
