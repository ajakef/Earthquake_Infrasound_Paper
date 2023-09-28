import numpy as np
import obspy
import pickle, pygmt, geopandas
import matplotlib.pyplot as plt
import pandas as pd
import cleanbf
from cleanbf import wyorm # colormap


#%% load local EQ results and set parameters
slowness_threshold = 2
try:
    with open('data/pkl/clean_aftershock_full.pkl', 'rb') as file:
        output = pickle.load(file)['output']
except:
    print()
    print('Could not find data/pkl/clean_aftershock_full.pkl. Are you currently in the right folder?')
    print()
    raise
            
sh = output['sh']
baz = output['back_az']
spec_4 = output['clean_polar_back']
r = 1e-5
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])
ac_max = np.quantile(spec_baz, 1)
event = obspy.UTCDateTime('2020-04-14T03:27:06')
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9') # transition between primary-secondary sound
loop_start = t_trans-10
spec_baz_times = output['t'] - (event - loop_start)

lon_station = -115.01833
lat_station = 44.27500

## latitude and longitude from relocation analysis
lon_eq = -115.01767
lat_eq = 44.26900
obspy.geodetics.gps2dist_azimuth(lat1=lat_station, lon1=lon_station, lat2=lat_eq, lon2=lon_eq) # 669 m S of PARK
z_eq = 7530 # depth below ground surface
ca_max = 328 # sound speed from data logger temperature records shows average temp at time of aftershock is -3C, so sound speed is 329
ca_min = 330
cs_max = 5600 # seismic wave speed ranges taken from Johnson et al., 2020 (GRL)
cs_min = 1400


#%% backproject local earthquake data and plot results 
lon_grid_range = [lon_station - 0.215, lon_station + 0.185]
lat_grid_range = [lat_station - 0.15, lat_station + 0.2]
grid_spacing_deg = 0.005
topo_local = pygmt.datasets.load_earth_relief('03s', region = (lon_grid_range[0], lon_grid_range[1], lat_grid_range[0], lat_grid_range[1])) # 3-arcsecond topo is fine at this scale. plotting through plt interface instead of pygmt interface is easier here because it is turning out to be very difficult to plot image overlays in pygmt and very easy in plt.

time_arrival = []
baz_arrival = []
power = []
for i in range(spec_baz.shape[0]):
    for j in range(spec_baz.shape[1]):
        if (spec_baz[i,j] > 1):
            time_arrival.append(spec_baz_times[i])
            baz_arrival.append(baz[j])
            power.append(spec_baz[i,j])
result, lons, lats, *_ = cleanbf.backproject(lon_grid_range, lat_grid_range, grid_spacing_deg, 
                                       lon_station, lat_station, lon_eq, lat_eq, z_eq, 
                                       ca_min, ca_max, cs_min, cs_max, 
                                       baz_arrival = baz_arrival, time_arrival = time_arrival, 
                                       power_arrival = np.array(power) * np.array(time_arrival), az_bin_width = 10)

fig, ax = plt.subplots()
a=ax.contourf(lons[1:], lats[1:], np.log10(100+result.T), cmap = wyorm, alpha = 1, levels = 50)
b=ax.contourf(topo_local.lon, topo_local.lat, topo_local.data, levels = 50, cmap = 'terrain', alpha = 0.35, vmin = 800)
ax.contour(topo_local.lon, topo_local.lat, topo_local.data, levels = 10, colors = 'black', alpha = 0.25)
plt.contour(lons[1:], lats[1:], result.T>400, colors = 'red', levels = 0, linewidths = 0.75)
ax.plot(lon_station, lat_station, 'b^', markersize = 7, label = 'PARK & Epicenter')
ax.plot(np.nan, 'r-', label = 'Source region', linewidth = 0.75)
deg_per_km = 360/(40e3 * np.cos(lat_station*np.pi/180))
ax.plot([-115, -115 + 10*deg_per_km], [44.15, 44.15], 'k-', linewidth = 3) # scale bar
ax.text(-115 + 10*deg_per_km, 44.15, '10 km', ha = 'center', va = 'bottom')
ax.legend()
ax.set_aspect(1/np.cos(lat_station * np.pi/180))
fig.colorbar(a, ax = ax, location = 'top', ticks = 10**np.arange(3, 6), shrink = 0.35, label = 'Infrasound Production')
fig.colorbar(b, ax = ax, location = 'right', ticks = np.arange(1500, 3001, 500), label = 'Elevation (m)')
fig.savefig('figures/Fig6a_aftershock_backprojection_map.png', dpi = 300)


#%% download seismic data for the Monte Cristo EQ backprojection plot
from obspy.clients.fdsn.mass_downloader import GlobalDomain, Restrictions, MassDownloader

## change project_dir to match where you store these files
domain = GlobalDomain()

t1 = obspy.UTCDateTime(2020, 5, 15, 11, 00, 00)
t2 = t1 + 3600

restrictions = Restrictions(
    # Get data from 5 minutes before the event to one hour after the
    # event. This defines the temporal bounds of the waveform data.
    starttime=t1,
    endtime=t2,
    # disable the restriction on distance between stations
    minimum_interstation_distance_in_m=0,
    network='BK,US,CC,IW,UO,N4,NN,UU',
    station='CMB,ELK,DLMT,DUG,TMBU,AHID,HAWA,MSO,RLMT,NEW,WOOD,O20A,FXWY,BEK,BW06,RDMU', # RDMU would be good too (NE UT) but it has serious square wave noise
    location=',??',
    channel="??Z"
)
mdl = MassDownloader(providers=['IRIS'])

# The data will be downloaded to the ``./waveforms/`` and ``./stations/``
# folders with automatically chosen file names.
mdl.download(domain, restrictions, mseed_storage="data/downloads/waveforms", stationxml_storage="data/downloads/stations")
#%% read in the seismic data for the Monte Cristo backprojection plot, and deconvolve response
station_info_all = pd.read_csv('data/seismic/2020-05-15-mww65-nevada.txt', sep = '|')
station_info_all_15 = station_info_all.loc[station_info_all.Distance < 15,:]


st = obspy.read('data/downloads/waveforms/*')
st = st.select(channel='BHZ') + st.select(station = 'WOOD', channel = 'HHZ') \
        + st.select(station='O20A', channel = 'HHZ') + st.select(station='BEK', channel = 'HHZ')\
            + st.select(station='RDMU', channel = 'HHZ')
bad_stations = ['ANWB', 'POHA']
st = obspy.Stream([tr for tr in st if (tr.stats.station not in bad_stations) and (tr.stats.sampling_rate >= 20) and (tr.stats.npts > 20000)])
st_stations = np.unique([tr.stats.station for tr in st if any(tr.stats.station == station_info_all_15.Station)])

## Just use certain stations verified to have good data
stations_to_use = ['ELK', 'AHID', 'DLMT', 'HAWA', 'DUG', 'NEW', 'CMB', 'RLMT', 'MSO', 'TMBU', 'WOOD', 'O20A', 'FXWY', 'BEK', 'BW06', 'RDMU']
st = obspy.Stream([st.select(station=station)[0] for station in stations_to_use])
station_info = station_info_all.loc[station_info_all_15.Station.isin(stations_to_use),:].sort_values('Distance').reset_index()

## deconvolve response and filter to 1-20 Hz band (matching CLEAN beamforming)
inv = obspy.read_inventory('data/downloads/stations/*')
for tr in st:
    print(tr)
    tr.remove_response(inv)
st.filter('bandpass', freqmin = 1, freqmax = 20)
eq_time = obspy.UTCDateTime('2020-05-15T11:03:27')
st.trim(eq_time, eq_time + 1000)
#%%  Monte Cristo: find seismic speed limits empirically
# Considering the 15 stations plotted in the Monte Cristo map with distances of about 2-10 degrees,
# find the speeds for the arrival times of the first 1% of energy (c_max) and first 90% of energy (c_min)

plt.figure()
for tr in st:
    j = np.where(station_info.Station == tr.stats.station)[0][0]
    p = np.cumsum(tr.data**2)
    p /= p.max()
    station_info.loc[j,'t1'] = tr.stats.delta*np.where(p > 0.01)[0][0]
    station_info.loc[j,'t2'] = tr.stats.delta*np.where(p > 0.9)[0][0]
    plt.subplot(len(st),1,1+j)
    if True:
        plt.plot(np.arange(tr.stats.npts) * tr.stats.delta, tr.data)
        plt.axvline(station_info.loc[j,'t1'], color = 'red')
        plt.axvline(station_info.loc[j,'t2'], color = 'red')
        plt.text(0,0,tr.stats.station)
    station_info.loc[j,'Distance'] = float(station_info.loc[j,'Distance'])
    station_info.loc[j,'cmin'] = station_info.loc[j,'Distance'] / (station_info.loc[j,'t2']) * 40e6/360
    station_info.loc[j, 'cmax'] = station_info.loc[j, 'Distance'] / (station_info.loc[j,'t1']) * 40e6/360

# plot results (not used in paper)
plt.figure()
plt.plot(station_info.Distance, station_info.cmin, 'ro')
plt.plot(station_info.Distance, station_info.cmax, 'bo')

# At 15/16 stations, first 1% of energy arrives faster than 2600 m/s.
# At 15/16 stations, first 90% of energy arrives slower than 7000 m/s.
cs_max = 7000 # max seismic speed from analysis above
cs_min = 2600 # min seismic speed from analysis above

#%% Plot cs_max and cs_min over actual seismograms to verify that they're reasonable estimates
plt.figure(figsize=(10,3.5))
plt.plot()
plt.ylim([10.5, 1.5])
station_info = station_info.sort_values('Distance')
for i, station in enumerate(station_info.Station):
    distance_m = station_info.Distance[i] * 40e6/360
    tr = st.select(station = station)[0]
    # To avoid clutter, skip a station if the difference in distance < 0.75 deg.
    if (i == 0) or ((station_info.Distance[i] - station_info.Distance[last_plotted]) > 0.75):
        last_plotted = i
        t = np.arange(tr.stats.npts) * tr.stats.delta
        plt.plot(t, tr.data/np.abs(tr.data).max() * 0.5 + station_info.Distance[i], 'k-')
        plt.text(450, -0.1 +station_info.Distance[i], f'{station_info.Net[i]}.{station_info.Station[i]}')

plt.plot([0, 11 * 40e6/360 /cs_min], [0, 11], linestyle = '--', color = 'gray')
plt.plot([0, 11 * 40e6/360 /cs_max], [0, 11], linestyle = '--', color = 'gray')
plt.xlim([0, 500])
plt.ylim([10.5, 0])
plt.xlabel('Time After Monte Cristo Earthquake (s)')
plt.ylabel('Epicentral Distance (degrees)')
plt.tight_layout()

plt.savefig('figures/Fig6c_MonteCristo_seismograms.png', dpi = 300)
#%% set up Monte Cristo backprojection
slowness_threshold = 2
with open('data/pkl/clean_MonteCristo_full.pkl', 'rb') as file:
    output = pickle.load(file)['output']

sh = output['sh'] 
baz = output['back_az']
spec_4 = output['clean_polar_back']
r = 1e-5
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])
## calculate outliers above background noise
mean = np.einsum('ij->j', spec_baz)/spec_baz.shape[0]
std = np.sqrt(np.einsum('ij->j', (spec_baz - mean)**2)/spec_baz.shape[0])
spec_baz[spec_baz < (mean + 2*std)] = 0

ac_max = np.quantile(spec_baz, 1)
spec_baz_times = output['t'] - (3*60 + 27) # eq happened 00:03:27 after analysis start

## find average atmospheric sound speed. Metadata shows PARK temp during Tonopah EQ was close to 
## 0 C, so 331 m/s at surface. But average sound speed could be somewhat more or less than that.
## cs_min and cs_max are defined above with the seismic data.
ca_max = 340 #  340 m/s corresponds to 14 C
ca_min = 320 # -20 C

## location of PARK infrasound station
lon_station = -115.01833
lat_station = 44.27500

## location of earthquake
lat_eq = 38.169
lon_eq = -117.850
print(obspy.geodetics.gps2dist_azimuth(lat1=lat_station, lon1=lon_station, lat2=lat_eq, lon2=lon_eq)) # 718 km, back-azimuth 200.24 deg
z_eq = 2700 # from USGS website, probably insignificant effect on propagation times
    

#%% set up Monte Cristo topo
lon_grid_range = [-125, -104.5]
lat_grid_range = [36, 49] # no sound comes from south of 36 or north of 48
## 30-arcsecond topo is good for regional scale. 
## Plotting through plt interface instead of pygmt interface is easier here because it is turning 
## out to be very difficult to plot image overlays in pygmt and very easy in plt.
topo_regional = pygmt.datasets.load_earth_relief('30s', region = (lon_grid_range[0], lon_grid_range[1], 
                                                                  lat_grid_range[0], lat_grid_range[1])) 

## To improve plot clarity, set ocean to a constant -1000 elevation. First, some below-sea-level 
## areas around California Delta need to be adjusted so they don't appear to be ocean. Due to xarray 
## indexing being weird, this is not easy to vectorize, but it only runs for a few seconds.
for i, lat in enumerate(np.array(topo_regional.lat)):
    for j, lon in enumerate(np.array(topo_regional.lon)):
        if (topo_regional.data[i,j] < 1) and (lat > 37.7) and (lon > -122.2):
            topo_regional.data[i,j] = 1
topo_regional.data[np.where(topo_regional.data < 0)] = -1000

#%% run the Monte Cristo backprojection
time_arrival = []
baz_arrival = []
power = []
for i in range(spec_baz.shape[0]):
    for j in range(spec_baz.shape[1]):
        if (spec_baz[i,j] > 1) and (spec_baz_times[i] < 2500) and (spec_baz_times[i] > 1000): # something (maybe plane) just before 900, plane after 2500 #### FIX THIS
            time_arrival.append(spec_baz_times[i])
            baz_arrival.append(baz[j])
            power.append(spec_baz[i,j])
grid_spacing_deg = 0.1
result,lons,lats,t_min,t_max=cleanbf.backproject(lon_grid_range, lat_grid_range, grid_spacing_deg, 
                                                 lon_station, lat_station, lon_eq, lat_eq, z_eq, 
                                                 ca_min, ca_max, cs_min, cs_max, 
                                                 baz_arrival=baz_arrival, time_arrival=time_arrival, 
                                                 power_arrival = np.array(power)*np.array(time_arrival), 
                                                 az_bin_width = 10)


#%% plot the regional backprojections and topography
fig, ax = plt.subplots()

## plot backprojections
a=ax.contourf(lons[1:], lats[1:], np.log10(10+result.T), cmap = wyorm, alpha = 1, levels = 50)
plt.contour(lons[1:], lats[1:], result.T>400, colors = 'red', levels = 0, linewidths = 0.75)

## plot topo
b=ax.contourf(topo_regional.lon, topo_regional.lat, 400+topo_regional.data, levels = 50, cmap = 'terrain', alpha = 0.25, vmin = -500, vmax = 4000)
ax.contour(topo_regional.lon, topo_regional.lat, topo_regional.data, levels = 8, colors = 'black', alpha = 0.25)

## set aspect ratio and plot scale bar

scale_lon = -124.75
scale_lat = 36.5
ax.set_aspect(1/np.cos(scale_lat * np.pi/180))
deg_per_km = 360/(40e3 * np.cos(scale_lat*np.pi/180))
ax.plot([scale_lon, scale_lon + 200*deg_per_km], [scale_lat, scale_lat], 'k-', linewidth = 3) # scale bar
ax.text(scale_lon + 0.5*200*deg_per_km, scale_lat, '200 km', ha = 'center', va = 'bottom', fontsize = 'small')

## plot nearest and most distant places that could make detectable infrasound between 1000-2500 s
plt.contour(lons, lats, t_max.T > 1000, levels = 0, colors = 'black', linewidths = 0.75, linestyles = 'dashed')
plt.contour(lons, lats, t_min.T > 2500, levels = 0, colors = 'black', linewidths = 0.75, linestyles = 'dashed')

## plot epicenter and station, and set up legend
ax.plot(lon_station, lat_station, 'b^', markersize = 7, label = 'PARK', markeredgecolor='white')
ax.plot(lon_eq, lat_eq, 'b*', markersize = 12, label = 'Epicenter', markeredgecolor='white')
ax.plot(np.nan, 'r-', label = 'Source', linewidth = 0.75)
ax.plot(np.nan, 'k.', label = 'Seismometer', markersize=10)
ax.legend(loc='upper right', borderaxespad=0, edgecolor='black', fontsize = 'small', labelspacing = 0.2)

## draw state lines
states = geopandas.read_file('data/state_lines/usa-states-census-2014.shp')
states.boundary.plot(ax = ax, color = 'black', linewidth = 0.5)

## set up colorbar and axes
cbt = fig.colorbar(b, ax = ax, location = 'right', ticks = np.arange(500, 4501, 500), label = 'Elevation (m)', shrink = 0.85)
cbt.ax.set_yticklabels([0,500,1000,1500,2000,2500,3000,3500,4000])
ax.set_xlim(lon_grid_range)
ax.set_ylim(lat_grid_range)
ax.set_xticks(np.arange(-125, -105.1, 5))
ax.set_yticks(np.arange(36,49,4))

## plot seismic station amplitudes
station_info['rms'] = [st.select(station = station)[0].std() for station in station_info.Station]
for i, station in enumerate(station_info.Station):
    print((station_info.Longitude[i], station_info.Latitude[i], station_info.rms[i]**2 * 1e10))
    plt.scatter(station_info.Longitude[i], station_info.Latitude[i], s = station_info.rms[i] * 2e6, c = 'black')
    plt.text(station_info.Longitude[i], station_info.Latitude[i]+station_info.rms[i]*0.6e4, station_info.Station[i], size = 'small')


fig.savefig('figures/Fig6b_MonteCristo_backprojection_map.png', dpi = 300)
