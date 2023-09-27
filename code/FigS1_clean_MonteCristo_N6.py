import numpy as np
import matplotlib.pyplot as plt
import obspy
import pickle, time
import cleanbf
from matplotlib import gridspec
import pandas as pd
from cleanbf import wyorm


pkl_file = 'data/pkl/clean_MonteCristo_full.pkl'
pkl_file3 = 'data/pkl/clean_MonteCristo_3.pkl'
pkl_file6 = 'data/pkl/clean_MonteCristo_6.pkl'

def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def image_trim_log(x): return(np.log(trim(x, ac_max, ac_max*r)))
def image_show_outliers(x, n=2):
    mean = np.einsum('ij->j', x)/x.shape[0]
    std = np.sqrt(np.einsum('ij->j', (x - mean)**2)/x.shape[0])
    x[x < (mean + n*std)] = 0
    return np.log(trim(x, ac_max, r*ac_max))

def slowness_image(M, lim, zlim = [0, 1]):
    plt.imshow(M.transpose(), extent = [-lim, lim, -lim, lim], origin='lower', vmin=zlim[0], vmax = zlim[1], cmap = wyorm)
    return plt.colorbar()

def circle(r):
    # run plt.plot(*circle(r))
    theta = np.arange(361) * np.pi/180
    return (r*np.sin(theta), r*np.cos(theta))
#%% set clean parameters (used in both the full array and sub-array calculations)

## define slowness grid to search
s_list = np.arange(-4, 4.01, 0.1)

## plot parameter
slowness_threshold = 2
baz_ticks = [-180, -90, 0, 90, 180]

loop_step = 4
loop_width = 4
phi = 0.1
win_length_sec = 1
freq_bin_width = 1
freq_min = 1
freq_max = 20
#%%
sub_array_3 = np.array([17, 9, 14])-1
sub_array_6 = np.array([17,9,14,18,5,7])-1
gs = gridspec.GridSpec(5, 3)

## clean parameters to use
s_list = np.arange(-4, 4, 0.1)

inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
coords_full = cleanbf.get_coordinates(inv)
coords_sub_3 = coords_full.iloc[sub_array_3,:]
coords_sub_6 = coords_full.iloc[sub_array_6,:]

slowness_threshold = 2
#%% Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)

eq_stream = obspy.read('data/MonteCristo.mseed')

omit = ['06'] # 06 is SN 084, which is noisy

eq_stream = obspy.Stream([ tr for tr in eq_stream if tr.stats.location not in omit])
eq_stream.filter('bandpass', freqmin = freq_min, freqmax = freq_max, corners = 4)

inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

## create a stream just with data for a 3-element subset of the full array
include = ['17', '09', '14', '18', '05', '07']
eq_stream_subset_6 = obspy.Stream([ tr for tr in eq_stream if tr.stats.location in include])

## define significant times in the data
event = obspy.UTCDateTime('2020-05-15T11:03:27') # https://earthquake.usgs.gov/earthquakes/eventpage/nn00725272/executive
trace_start = eq_stream[0].stats.starttime # start of trace
trace_end = eq_stream[0].stats.endtime # end of trace

loop_start = trace_start
loop_end = trace_end
#%% supplement: run the beamforming calculation for the 6-element sub-array
## create a stream just with data for a 3-element subset of the full array

calculate_new_beamform_result_6 = False # to save a lot of time, set this to False once you've already run the calculation

if calculate_new_beamform_result_6:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream_subset_6.slice(loop_start, loop_end), loop_width = loop_width, 
                              loop_step = loop_step, verbose = False, phi = phi, separate_freqs = 0, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                              sxList = s_list, syList = s_list, prewhiten = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file6, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file6, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)

#%% plot 6-station results
sh = output['sh']
baz = output['back_az']-180
spec_4 = output['clean_polar_back']
r = 1e-2
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:,ind]
ac_max = np.quantile(spec_baz, 0.999)


def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def image_trim_log(x): return(np.log(trim(x, ac_max, ac_max*r)))
def image_show_outliers(x):
    mean = np.einsum('ij->j', x)/x.shape[0]
    std = np.sqrt(np.einsum('ij->j', (x - mean)**2)/x.shape[0])
    x[x < (mean +2*std)] = 0
    return np.log(trim(x, ac_max, r*ac_max))

plt.subplot(5,1,1)
cleanbf.image(image_trim_log(spec_s), output['t']-(event - loop_start), sh, crosshairs = False)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks([0,1,2,3])
plt.title('A. N=6, Slowness (6 sensors)', loc = 'left', fontsize = 11)
w = output['original_sh'] < 4

plt.subplot(5,1,2)
cleanbf.image(image_trim_log(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.ylabel('degrees')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('B. N=6, Backazimuth (Slowness > 2 s/km)', loc = 'left')

plt.subplot(5,1,3)
im = cleanbf.image(image_show_outliers(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.ylabel('degrees')
# only plot x ticks for bottom panel to save space
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('C. N=6, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 11)
#%% N = 3
with open(pkl_file3, 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
sh = output['sh']
baz = output['back_az']-180
spec_4 = output['clean_polar_back']
r = 1e-2
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:,ind]
ac_max = np.quantile(spec_baz, 0.999)
    
plt.subplot(5,1,4)
cleanbf.image(image_show_outliers(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.ylabel('degrees')
# only plot x ticks for bottom panel to save space
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('D. N=3, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 11)
    
#%% full array
with open(pkl_file, 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
sh = output['sh']
baz = output['back_az']-180
spec_4 = output['clean_polar_back']
r = 1e-2
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:,ind]
ac_max = np.quantile(spec_baz, 0.999)
    
plt.subplot(5,1,5)
cleanbf.image(image_show_outliers(spec_baz), output['t']-(event - trace_start), baz, crosshairs = False)
plt.ylabel('degrees')#plt.xlabel('Time after earthquake (seconds)')
plt.xticks(np.arange(0, 3001, 500))
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('E. N=17, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 11)
plt.xlabel('Time after earthquake (seconds)')
plt.ylabel('degrees')

#%%
plt.gcf().set_size_inches( 6.5, 6.5, forward=True) 
plt.tight_layout()
#%%
fig = plt.gcf()
fig.subplots_adjust(right=0.91)
cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.5])
fig.colorbar(im,cax=cbar_ax, label='Infrasound Power', ticks = [])
#%%
plt.savefig('figures/FigS1_montecristo_17_6_paper.png')
