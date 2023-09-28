import numpy as np
import matplotlib.pyplot as plt
import obspy
import pickle, time
import cleanbf

pkl_file = 'data/pkl/clean_MonteCristo_full.pkl'
pkl_file3 = 'data/pkl/clean_MonteCristo_3.pkl'

def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def image_trim_log(x): return(np.log(trim(x, ac_max, ac_max*r)))
def image_show_outliers(x, n=2):
    mean = np.einsum('ij->j', x)/x.shape[0]
    std = np.sqrt(np.einsum('ij->j', (x - mean)**2)/x.shape[0])
    x[x < (mean + n*std)] = 0
    return np.log(trim(x, ac_max, r*ac_max))

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
#%% Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)

eq_stream = obspy.read('data/MonteCristo.mseed')

omit = ['06'] # 06 is SN 084, which is noisy

eq_stream = obspy.Stream([ tr for tr in eq_stream if tr.stats.location not in omit])
eq_stream.filter('bandpass', freqmin = freq_min, freqmax = freq_max, corners = 4)

inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

## create a stream just with data for a 3-element subset of the full array
include = ['17', '09', '14']
eq_stream_subset_3 = obspy.Stream([ tr for tr in eq_stream if tr.stats.location in include])

## define significant times in the data
event = obspy.UTCDateTime('2020-05-15T11:03:27') # https://earthquake.usgs.gov/earthquakes/eventpage/nn00725272/executive
trace_start = eq_stream[0].stats.starttime # start of trace
trace_end = eq_stream[0].stats.endtime # end of trace

loop_start = trace_start
loop_end = trace_end
#%% plot waveform
plt.close('all')
plt.subplot(7,1,1)
skip=500 # skip the beginning to avoid filter artifacts. not noticeable at this plot scale.
gem_bitweight = 3.5012e-3 # mPa/count
start_offset = event - trace_start
plt.plot(np.arange(len(eq_stream[0])-skip) * 0.01 - start_offset, gem_bitweight * eq_stream[0].data[skip:], 'k-', linewidth=0.5)
# set_xlim leaves some padding; axis('image') changes the aspect ratio. This sets tight axis limits without forcing a bad aspect ratio.
plt.gca().set_xbound(-start_offset,3600-start_offset) 
plt.ylabel('Pa')
plt.title('a. Infrasound Waveform', loc = 'left', fontsize = 'small')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(np.arange(-1,2)*0.025)

#%% run the beamforming calculation for full array
calculate_new_beamform_result = False # to save a TON of time, set this to False once you've already run the calculation
if calculate_new_beamform_result:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream.slice(loop_start, loop_end), loop_width = loop_width, 
                              loop_step = loop_step, separate_freqs = 0, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                              sxList = s_list, syList = s_list, prewhiten = False, phi = 0.1, verbose = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file, 'wb') as file:
        pickle.dump({'output':output}, file)
    output_full = output
else:
    with open(pkl_file, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)

#%% plot results for full array
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


plt.subplot(7,1,2)
cleanbf.image(image_trim_log(spec_s), output['t']-(event - trace_start), sh, crosshairs = False)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks([0,1,2,3])
plt.title('b. Power vs. Time, Slowness', loc = 'left', fontsize = 'small')
w = output['original_sh'] < 4

plt.subplot(7,1,3)
cleanbf.image(image_trim_log(spec_baz), output['t']-(event - trace_start), baz, crosshairs = False)
plt.ylabel('degrees')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('c. Power vs. Time, Backazimuth (Slowness > 2 s/km)', loc = 'left', fontsize = 'small')

plt.subplot(7,1,4)
cleanbf.image(image_show_outliers(spec_baz), output['t']-(event - trace_start), baz, crosshairs = False)
plt.ylabel('degrees')#plt.xlabel('Time after earthquake (seconds)')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('d. Power vs. Time, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 'small')

plt.tight_layout()

#%% run the beamforming calculation for just the 3-element sub-array
calculate_new_beamform_result_3 = False # to save a lot of time, set this to False once you've already run the calculation

if calculate_new_beamform_result_3:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream_subset_3.slice(loop_start, loop_end), loop_width = loop_width, 
                              loop_step = loop_step, verbose = False, phi = phi, separate_freqs = 0, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                              sxList = s_list, syList = s_list, prewhiten = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file3, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file3, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)

#%% plot 3-station results

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

plt.subplot(7,1,5)
cleanbf.image(image_trim_log(spec_s), output['t']-(event - loop_start), sh, crosshairs = False)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks([0,1,2,3])
plt.title('e. Power vs. Time, Slowness (3 sensors)', loc = 'left', fontsize = 'small')
w = output['original_sh'] < 4

plt.subplot(7,1,6)
cleanbf.image(image_trim_log(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.ylabel('degrees')
plt.xticks(np.arange(0, 3001, 500), labels = [])
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('f. Power vs. Time, Backazimuth (Slowness > 2 s/km)', loc = 'left', fontsize = 'small')

plt.subplot(7,1,7)
im=cleanbf.image(image_show_outliers(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.xlabel('Time after earthquake (seconds)')
plt.ylabel('degrees')
# only plot x ticks for bottom panel to save space
plt.xticks(np.arange(0, 3001, 500))
plt.yticks(baz_ticks)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
plt.title('g. Power vs. Time, Backazimuth (Slowness > 2 s/km; above-ambient)', loc = 'left', fontsize = 'small')


#%%
fig = plt.gcf()
fig.set_size_inches(6.5, 9, forward=True) # max fig size for villarrica screen: 12.94x6.65 inch
fig.tight_layout()
cbar_ax = fig.add_axes([0.93, 0.15, 0.02, 0.7])
fig.colorbar(im,cax=cbar_ax, label='Infrasound Power', ticks = [])
fig.subplots_adjust(right=0.91)



#%%
plt.savefig('figures/Fig5_montecristo_17_3_paper.png', dpi = 300)

