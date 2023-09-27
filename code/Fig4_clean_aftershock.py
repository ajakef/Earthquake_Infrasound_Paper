import numpy as np
import matplotlib.pyplot as plt
import obspy
import pickle, time
import cleanbf

#%% define parameters
pkl_file = 'data/pkl/clean_aftershock_full.pkl'
pkl_file3 = 'data/pkl/clean_aftershock_3.pkl'
pkl_file6 = 'data/pkl/clean_aftershock_6.pkl'


t1 = obspy.UTCDateTime('2020-04-14T03:25:45.600000Z')
t2 = obspy.UTCDateTime('2020-04-14T03:28:27.600000Z')
event = obspy.UTCDateTime('2020-04-14T03:27:06')
t_trans = obspy.UTCDateTime('2020-04-14T03:27:08.9') # transition between primary-secondary sound

## define slowness grid to search
s_list = np.arange(-4, 4.01, 0.1)

## plot parameters
slowness_threshold = 2
xticks = np.arange(0, 70.1, 10)
baz_ticks = [-180, -90, 0, 90, 180]

## loop parameters
loop_start = t_trans-10
loop_end = t_trans + 75
loop_step = 1
loop_width = 2
phi = 0.1
win_length_sec = 1
freq_min = 2
freq_max = 25
separate_freqs = 0
freq_bin_width = 1




#%% Load an earthquake recording. This includes preliminary background noise,
## primary infrasound (simple wavefield), and secondary infrasound (diffuse wavefield)
## M3.5, 2020-04-14 03:27:06 44.284°N 115.029°W 10.0 km depth
## https://earthquake.usgs.gov/earthquakes/eventpage/us70008vuc/executive
eq_stream = obspy.read('data/aftershock.mseed') # omits 06 (084, noisy) and 22 (090, early dropout)

eq_stream.filter('bandpass', freqmin=freq_min, freqmax = freq_max, corners=4)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream, inv) # store the coordinates in the stream

#%% plot waveform
plt.close('all')
plt.subplot(7,1,1)
skip=500 # skip the beginning to avoid filter artifacts. not noticeable at this plot scale.
gem_bitweight = 3.5012e-3 # mPa/count

t = np.arange((loop_start - event + loop_width/2)*100, (loop_end - event - loop_width/2)*100 + 1) * 0.01
plt.plot(t, gem_bitweight * eq_stream[0].slice(loop_start+loop_width/2, loop_end-loop_width/2).data, 'k-', linewidth=0.5)

## plot an inset of 3 traces after amplitude drops off, blown up
include = ['09', '14', '17']
inset_stream = eq_stream.copy()
inset_stream = obspy.Stream([ tr for tr in inset_stream if tr.stats.location in include])
inset_stream.filter('highpass', freq=15)
inset_indices = np.where((t > 30) & (t < 100))[0]
for i in range(0, 3):
    print(inset_stream[i])
    y = gem_bitweight * inset_stream[i].slice(loop_start+loop_width/2, loop_end-loop_width/2).data
    plt.plot(t[inset_indices], -(i+0.5)/1.4 + 30 * y[inset_indices], 'k-', linewidth=0.5)
# set_xlim leaves some padding; axis('image') changes the aspect ratio. This sets tight axis limits without forcing a bad aspect ratio.
plt.gca().set_xbound(t[0],t[-1])
plt.ylabel('Pa')
plt.title('A. Infrasound Waveform', loc = 'left', fontsize = 11)
plt.xticks(xticks, labels = [])
plt.yticks([-2, 0, 2])


#%% run the beamforming calculation for 20 sensors
calculate_new_beamform_result = False # To save time, only set as True if you want to re-do beamforming, not use precalculated result
if calculate_new_beamform_result:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream.slice(loop_start, loop_end), loop_width = loop_width, 
                              loop_step = loop_step, verbose = False, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                              sxList = s_list, syList = s_list, prewhiten = False, phi = phi)
    output['processing_time'] = time.time() - analysis_start_time
    
    with open(pkl_file, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)
output_full = output
#%% Run obspy's traditional beamformer for runtime comparison (not needed for making figures)
if False:
    analysis_start_time = time.time()
    array_proc_output = obspy.signal.array_analysis.array_processing(eq_stream.slice(loop_start, loop_end), loop_width, loop_step/loop_width, 
                                                 s_list.min(), s_list.max(), s_list.min(), s_list.max(), np.diff(s_list)[0],
                                                 0, 0, freq_min, freq_max, loop_start, loop_end, prewhiten = False)
    print( time.time() - analysis_start_time) # 15.6 vs 3367 for full array CLEAN

#%% plot results for 20 sensors
sh = output['sh']
baz = output['back_az'] - 180
spec_4 = output['clean_polar_back']
r = 1e-5
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:,ind]
ac_max = np.quantile(spec_baz, 1)

def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def imageAdj(x): return(np.log(trim(x, ac_max, ac_max*r)))
plt.subplot(7,1,2)
cleanbf.image(np.log(trim(spec_s, spec_s[:,w].max(), r*spec_s[:,w].max())), output['t']-(event - loop_start), sh, crosshairs = False)
#plt.axvline(0, color = 'black', lw = 0.5)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.title('B. Power vs. Time, Slowness', loc = 'left')
w = output['original_sh'] < 4
plt.plot(output['t'][w]-(event - loop_start), output['original_sh'][w], 'k.', markersize=5)
plt.plot(output['t'][w]-(event - loop_start), output['original_sh'][w], 'w.', markersize=2)
plt.yticks([0,1,2,3])
plt.xticks(xticks, labels = [])
plt.gca().set_xbound(t[0],t[-1]) 


plt.subplot(7,1,3)
cleanbf.image(imageAdj(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False)
plt.ylabel('degrees')
#plt.xlabel('Time after earthquake (seconds)')
plt.title(f'C. Power vs Time, Backazimuth (Slowness > {slowness_threshold} s/km)', loc = 'left')
plt.plot(output['t']-(event - loop_start), (output['original_az']) % 360 - 180, 'k.', markersize=5)
plt.plot(output['t']-(event - loop_start), (output['original_az']) % 360 - 180, 'w.', markersize=2)
plt.yticks(baz_ticks)
plt.xticks(xticks, labels = [])

plt.margins(0.01)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
#plt.axvline(0, color = 'black', lw = 0.5)
plt.gca().set_xbound(t[0],t[-1]) 
plt.tight_layout()
plt.subplots_adjust(hspace = 0.5)


#%% calculate 3-station clean result

calculate_new_beamform_result_3 = False # set to False after running the calculation once to save time

include = ['09', '14', '17']
eq_stream_3 = obspy.read('data/aftershock.mseed')
eq_stream_3 = obspy.Stream([ tr for tr in eq_stream if tr.stats.location in include])
eq_stream_3.filter('bandpass', freqmin=freq_min, freqmax = freq_max, corners=4)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream_3, inv) # store the coordinates in the stream

if calculate_new_beamform_result_3:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream_3.slice(loop_start, loop_end), loop_width = loop_width, loop_step = loop_step, 
                              verbose = False, phi = phi, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, # formerly freq 3-25
                              sxList = s_list, syList = s_list, prewhiten = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file3, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file3, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)
output_3 = output        
#%% Run obspy's traditional beamformer for runtime comparison (not needed for making figures)
if False:
    analysis_start_time = time.time()
    array_proc_output=obspy.signal.array_analysis.array_processing(eq_stream_3.slice(loop_start, loop_end), loop_width, loop_step/loop_width, 
                                                 s_list.min(), s_list.max(), s_list.min(), s_list.max(), np.diff(s_list)[0],
                                                 0, 0, freq_min, freq_max, loop_start, loop_end, prewhiten = False)
    print( time.time() - analysis_start_time) # about 0.6 seconds, compared to 58 seconds for sub-array CLEAN 


#%% plot 3-station clean result
sh = output['sh']
baz = output['back_az'] - 180
spec_4 = output['clean_polar_back']
r = 1e-5
w = (sh > slowness_threshold)
spec_s = np.einsum('hijk->hk', spec_4)
spec_f = np.einsum('hijk->hi', spec_4)
ind = np.concatenate([np.arange(18,36), np.arange(18)]) # rearrange to -180 to 180 range
spec_baz = np.einsum('hijk->hj', spec_4[:,:,:,w])[:, ind]
ac_max = np.quantile(spec_baz, 1)

def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def imageAdj(x): return(np.log(trim(x, ac_max, ac_max*r)))
plt.subplot(7,1,4)
cleanbf.image(np.log(trim(spec_s, spec_s[:,w].max(), r*spec_s[:,w].max())), output['t']-(event - loop_start), sh, crosshairs = False)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.title('D. Power vs. Time, Slowness (3 sensors)', loc = 'left')
w = output['original_sh'] < 4
plt.plot(output['t'][w]-(event - loop_start), output['original_sh'][w], 'k.', markersize=5)
plt.plot(output['t'][w]-(event - loop_start), output['original_sh'][w], 'w.', markersize=2)
plt.yticks([0,1,2,3])
plt.xticks(xticks, labels = [])
plt.gca().set_xbound(t[0],t[-1]) 

plt.subplot(7,1,5)
im = cleanbf.image(imageAdj(spec_baz), output['t']-(event - loop_start), baz, crosshairs = False) # im used later for colorbar
plt.ylabel('degrees')
#plt.xlabel('Time after earthquake (seconds)')
plt.title(f'E. Power vs. Time, Backazimuth (Slowness > {slowness_threshold} s/km; 3 sensors)', loc = 'left')
plt.plot(output['t']-(event - loop_start), (output['original_az']) % 360 - 180, 'k.', markersize=5)
plt.plot(output['t']-(event - loop_start), (output['original_az']) % 360 - 180, 'w.', markersize=2)
plt.yticks(baz_ticks)
plt.margins(0.01)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
#plt.axvline(0, color = 'black', lw = 0.5)
plt.gca().set_xbound(t[0],t[-1]) 
plt.tight_layout()
plt.subplots_adjust(hspace = 0.5)
#%% Plot traditional beamforming results of N=20 vs N=3
plt.subplot(7,1,6)
plt.axhline(slowness_threshold, color = 'black', lw = 0.5, ls = '--')
plt.ylabel('s/km')
plt.title('F. Slowness, traditional beamforming (3 sensors vs. 20 sensors)', loc = 'left')
w = output['original_sh'] < 4
plt.plot(output_full['t'][w]-(event - loop_start), output_full['original_sh'][w], 'b.', markersize=5)
plt.plot(output_3['t'][w]-(event - loop_start), output_3['original_sh'][w], 'r.', markersize=5)
plt.yticks([0,1,2,3])
plt.xticks(xticks, labels = [])
plt.gca().set_xbound(t[0],t[-1]) 

plt.subplot(7,1,7)
plt.ylabel('degrees')
plt.xlabel('Time after earthquake (seconds)')
plt.title('G. Backazimuth, traditional beamforming (3 sensors vs. 20 sensors)', loc = 'left')
plt.plot(output_full['t']-(event - loop_start), (output_full['original_az']) % 360 - 180, 'b.', markersize=5, label = 'N=20')
plt.plot(output_3['t']-(event - loop_start), (output_3['original_az']) % 360 - 180, 'r.', markersize=5, label = 'N=3')
plt.legend(loc='upper right', framealpha=1)
plt.yticks(baz_ticks)
plt.margins(0.01)
for i in baz_ticks: plt.axhline(i, color = 'gray', lw = 0.25)
#plt.axvline(0, color = 'black', lw = 0.5)
plt.gca().set_xbound(t[0],t[-1]) 
plt.tight_layout()
plt.subplots_adjust(hspace = 0.5)
#%% format figure nicely
fig = plt.gcf()
fig.set_size_inches(7.5, 10, forward=True) # max fig size for villarrica screen: 12.94x6.65 inch
fig.tight_layout()
cbar_ax = fig.add_axes([0.93, 0.325, 0.02, 0.5])
fig.colorbar(im,cax=cbar_ax, label='Infrasound Power', ticks = [])
fig.subplots_adjust(right=0.91)


#%% save figure
plt.savefig('figures/Fig4_Aftershock_20_3_paper.png')
#%% calculate 6-station clean result for use in supplementary figure 

calculate_new_beamform_result_6 = False # set to False after running the calculation once to save time

#include = ['09', '14', '17', '5', '7', '18'] # Anderson paper
include = ['01', '07', '12', '17', '19', '22'] # Scamfer paper

eq_stream_6 = obspy.read('data/aftershock.mseed')
eq_stream_6 = obspy.Stream([ tr for tr in eq_stream if tr.stats.location in include])
eq_stream_6.filter('bandpass', freqmin=freq_min, freqmax = freq_max, corners=4)
inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
cleanbf.add_inv_coords(eq_stream_6, inv) # store the coordinates in the stream

if calculate_new_beamform_result_6:
    analysis_start_time = time.time()
    output = cleanbf.clean_loop(eq_stream_6.slice(loop_start, loop_end), loop_width = loop_width, loop_step = loop_step, 
                              verbose = False, phi = phi, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                              freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, # formerly freq 3-25
                              sxList = s_list, syList = s_list, prewhiten = False)
    output['processing_time'] = time.time() - analysis_start_time
    with open(pkl_file6, 'wb') as file:
        pickle.dump({'output':output}, file)
else:
    with open(pkl_file6, 'rb') as file:
        d = pickle.load(file)
        locals().update(d)
output_6 = output     