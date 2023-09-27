import obspy, shutil
from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

## change project_dir to match where you store these files
domain = GlobalDomain()


#%% Download data for local event
t1 = obspy.UTCDateTime('2020-04-14T03:25:45.600000Z')
t2 = obspy.UTCDateTime('2020-04-14T03:28:27.600000Z')
restrictions = Restrictions(
    # Get data from 5 minutes before the event to one hour after the
    # event. This defines the temporal bounds of the waveform data.
    starttime=t1,
    endtime=t2+2, # a bit extra to allow time offset correction
    # disable the restriction on distance between stations
    minimum_interstation_distance_in_m=0,
    network='XP',
    station='PARK',
    location='??',
    channel="HDF"
)

## set up the downloader to send requests to IRIS-DMC
mdl = MassDownloader(providers=['IRIS'])

# The data will be downloaded to the ``./waveforms/`` and ``./stations/``
# folders with automatically chosen file names.
mdl.download(domain, restrictions, mseed_storage="data/downloads/waveforms", stationxml_storage="data/downloads/stations")

## Run this code in a try block in case the download failed
try:
    ## correct 2-second time offset on locations 12 and 20, and remove noisy location 06
    st_raw = obspy.read('data/downloads/waveforms/*20200414*')
    st_processed = obspy.Stream()
    for tr in st_raw:
        if tr.stats.location == '06':
            continue
        if tr.stats.location in ['12', '20']:
            tr.stats.starttime -= 2
        st_processed += tr
    st_processed.trim(t1, t2)
    assert len(st_processed) == 20 # make sure we downloaded all 20 traces
    st_processed.write('data/aftershock.mseed') # omits 06 (084, noisy) and 22 (090, early dropout)
    # copy the metadata to a more convenient place
    shutil.copyfile('data/downloads/stations/XP.PARK.xml', 'data/XP_PARK_inventory.xml')
except: 
    print('Failed to read downloaded files. Try downloading them again.')

#%% Download data for regional event (Monte Cristo)
t1 = obspy.UTCDateTime(2020, 5, 15, 11, 00, 00)
t2 = t1 + 3600

restrictions = Restrictions(
    # Get data from 5 minutes before the event to one hour after the
    # event. This defines the temporal bounds of the waveform data.
    starttime=t1,
    endtime=t2,
    # disable the restriction on distance between stations
    minimum_interstation_distance_in_m=0,
    network='XP',
    station='PARK',
    location='??',
    channel="HDF"
)

## set up the downloader to send requests to IRIS-DMC
mdl = MassDownloader(providers=['IRIS'])

# The data will be downloaded to the ``./waveforms/`` and ``./stations/``
# folders with automatically chosen file names.
mdl.download(domain, restrictions, mseed_storage="data/downloads/waveforms", stationxml_storage="data/downloads/stations")


## Run this code in a try block in case the download failed
try:
    st = obspy.read('data/downloads/waveforms/XP.PARK*20200515*')
    # Exclude the noisy location 06, then save all traces in a single miniSEED file
    st = obspy.Stream([ tr for tr in st if tr.stats.location != '06'])

    assert len(st) == 17 # make sure we downloaded all 17 traces
    st.write('data/MonteCristo.mseed')
    # copy the metadata to a more convenient place
    shutil.copyfile('data/downloads/stations/XP.PARK.xml', 'data/XP_PARK_inventory.xml')
except: 
    print('Failed to read downloaded files. Try downloading them again.')

