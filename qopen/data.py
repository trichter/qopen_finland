import numpy as np
from obspy import read

DATA2018 = '../data/2018_IMS/EVENT_DATA/{evid}/{evid}.mseed'
DATA2020 = '../data/2020_ISUH_FULLY_PICKED/EVENT_DATA/{evid}/{evid}.mseed'

import logging
log = logging.getLogger('data')

def get_data(network, station, location, channel, starttime, endtime, event):
    evid = str(event.resource_id).split('/')[-1]
    year = starttime.year
    if year == 2020 and network == 'OT':
        # OT station moved -> no use in monitoring application
        return
    data = DATA2020 if year == 2020 else DATA2018    
    fname = data.format(evid=evid)
    seedid = '.'.join([network, station, location, channel])
    try:
        stream = read(fname, 'MSEED', sourcename=seedid)
    except Exception as ex:
        print(ex)
        stream = []
    if len(stream) == 0:
        log.debug('no data for %s.%s', network, station)
        print('no data for %s.%s' % (network, station))
        return
    stream.trim(starttime, endtime)
    if station == 'MUNK':
        dataE = stream.select(component='E')[0].data
        dataN = stream.select(component='N')[0].data
        maxE = np.max(np.abs(dataE-np.mean(dataE)))
        maxN = np.max(np.abs(dataN-np.mean(dataN)))
        if  maxE > 50 * maxN:
            log.debug('%s: channel E of MUNK is not working -> load channel N instead', evid)
            print(evid, 'channel E of MUNK is not working -> load channel N instead')
            traces = []
            for tr in stream:
                if tr.stats.component == 'Z':
                    traces.append(tr)
                if tr.stats.component == 'N':
                    traces.append(tr)
                    tr2 = tr.copy()
                    tr2.stats.component = 'E'
                    traces.append(tr2)
            stream.traces = traces
    assert len(stream) == 3
    return stream
