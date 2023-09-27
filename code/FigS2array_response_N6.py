
import numpy as np
import matplotlib.pyplot as plt
import obspy
import pickle, time
import cleanbf
from matplotlib import gridspec
import pandas as pd
from cleanbf import wyorm


def trim(x, a, b): x[x>a]=a; x[x<b]=b; return x
def image_trim_log(x): return(np.log(trim(x, ac_max, ac_max*r)))
def image_show_outliers(x, n=2):
    mean = np.einsum('ij->j', x)/x.shape[0]
    std = np.sqrt(np.einsum('ij->j', (x - mean)**2)/x.shape[0])
    x[x < (mean + n*std)] = 0
    return np.log(trim(x, ac_max, r*ac_max))

def slowness_image(M, lim, zlim = [0, 1], cbar = True):
    plt.imshow(M.transpose(), extent = [-lim, lim, -lim, lim], origin='lower', vmin=zlim[0], vmax = zlim[1], cmap = wyorm)
    if cbar:
        return plt.colorbar()
    else:
        return

def circle(r):
    # run plt.plot(*circle(r))
    theta = np.arange(361) * np.pi/180
    return (r*np.sin(theta), r*np.cos(theta))

sub_array_3 = np.array([17, 9, 14])-1
sub_array_6 = np.array([17,9,14,18,5,7])-1 # Anderson paper
gs = gridspec.GridSpec(7, 3)

## clean parameters to use
s_list = np.arange(-4, 4, 0.1)

inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
coords_full = cleanbf.get_coordinates(inv)
coords_sub_3 = coords_full.iloc[sub_array_3,:]
coords_sub_6 = coords_full.iloc[sub_array_6,:]
#%% A: map, with 9, 14, and 17 marked
plt.subplot(gs[:3,:])
plt.scatter(1000*coords_sub_3.x, 1000*coords_sub_3.y, edgecolor = 'red', marker = '^', s = 70, label = 'Sub-array (N=3)')
plt.scatter(1000*coords_sub_6.x, 1000*coords_sub_6.y, facecolor = 'white', edgecolor = 'orange', marker = 'o', s = 30, label = 'Sub-array (N=6)')
plt.scatter(1000*coords_full.x, 1000*coords_full.y, color = 'black', marker = 'o', s = 7, label = 'Full Array (N=22)')

plt.axis('equal')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('A. Station Map for XP.PARK', loc='left', fontsize='medium')
plt.legend(loc = 'upper right', fontsize = 'small')
#%% B: 3-element array response (wavenumber)
plt.subplot(gs[3:5,0])
xy = np.append(np.array(coords_sub_3.loc[:,['x', 'y']]),np.zeros([3,1]), axis=1)

klim = 2*np.pi*10/0.330 # wavenumber at 10 Hz in rad/km
kstep = 1

arrayResp = obspy.signal.array_analysis.array_transff_wavenumber(xy, coordsys = 'xy', klim=klim, kstep=kstep)

slowness_image(arrayResp, klim/100, cbar = False)

plt.plot(*circle(2*np.pi * 2 / 0.333/100), 'k--')
plt.plot(*circle(2*np.pi * 10 / 0.333/100), 'k--')
plt.title('B. Array Response (N=3)', loc = 'left', fontsize='medium')
plt.xlabel('$k_x$ (×$10^2$ rad/km)', labelpad = -1)
plt.ylabel('$k_y$ (×$10^2$ rad/km)', labelpad = -1)

###############################
t_index = 14 # 14: multiple sonic waves. 9: seismic wave with aliasing.
plt.subplot(gs[5:,0])
with open('data/pkl/clean_aftershock_3.pkl', 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
    o_sub = output['original'][t_index,:,:]
    o_sub = o_sub/o_sub.max() * output['original_semblance'][t_index]

slowness_image(o_sub, s_list.max(), [o_sub.min(), o_sub.max()], cbar = False)
plt.plot(*circle(3), 'k--')
plt.xlabel('$s_x$ (s/km)', labelpad = -1)
plt.ylabel('$s_y$ (s/km)', labelpad = -1)
plt.title('C. Slowness Spectrum (N=3)', loc = 'left', fontsize = 'medium')
#%% C: 6-element array response (wavenumber)
plt.subplot(gs[3:5,1])
xy = np.append(np.array(coords_sub_6.loc[:,['x', 'y']]),np.zeros([6,1]), axis=1)

klim = 2*np.pi*10/0.330 # wavenumber at 10 Hz in rad/km
kstep = 1

arrayResp = obspy.signal.array_analysis.array_transff_wavenumber(xy, coordsys = 'xy', klim=klim, kstep=kstep)

slowness_image(arrayResp, klim/100, cbar = False)

plt.plot(*circle(2*np.pi * 2 / 0.333/100), 'k--')
plt.plot(*circle(2*np.pi * 10 / 0.333/100), 'k--')
plt.title('D. N=6', loc = 'left', fontsize='medium')
plt.xlabel('$k_x$ (×$10^2$ rad/km)', labelpad = -1)
plt.ylabel('$k_y$ (×$10^2$ rad/km)', labelpad = -1)

###############################
t_index = 14 # 14: multiple sonic waves. 9: seismic wave with aliasing.
plt.subplot(gs[5:,1])
with open('data/pkl/clean_aftershock_6.pkl', 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
    o_sub = output['original'][t_index,:,:]
    o_sub = o_sub/o_sub.max() * output['original_semblance'][t_index]

slowness_image(o_sub, s_list.max(), [o_sub.min(), o_sub.max()], cbar = False)
plt.plot(*circle(3), 'k--')
plt.xlabel('$s_x$ (s/km)', labelpad = -1)
plt.ylabel('$s_y$ (s/km)', labelpad = -1)
plt.title('E. N=6', loc = 'left', fontsize = 'medium')
#%% D: PARK response (wavenumber)
plt.subplot(gs[3:5,2])
xy = np.append(np.array(coords_full.loc[:,['x', 'y']]), np.zeros([22,1]), axis=1)

klim = 2*np.pi*10/0.330 # wavenumber at 10 Hz in rad/km
kstep = 1

arrayResp = obspy.signal.array_analysis.array_transff_wavenumber(xy, coordsys='xy', klim=klim, kstep=kstep)

slowness_image(arrayResp, klim/100, cbar = False)
#cbar.set_label('Semblance')
plt.plot(*circle(2*np.pi * 2 / 0.333/100), 'k--')
plt.plot(*circle(2*np.pi * 10 / 0.333/100), 'k--')
plt.title('F. Full Array', loc = 'left', fontsize='medium')
plt.xlabel('$k_x$ (×$10^2$ rad/km)', labelpad = -1)
plt.ylabel('$k_y$ (×$10^2$ rad/km)', labelpad = -1)
###############################

t_index = 14 # 14: multiple sonic waves. 9: seismic wave with aliasing.
plt.subplot(gs[5:,2])
with open('data/pkl/clean_aftershock_full.pkl', 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
    o_sub = output['original'][t_index,:,:]
    o_sub = o_sub/o_sub.max() * output['original_semblance'][t_index]

slowness_image(o_sub, s_list.max(), [o_sub.min(), o_sub.max()], cbar = False)
plt.plot(*circle(3), 'k--')
plt.xlabel('$s_x$ (s/km)', labelpad = -1)
plt.ylabel('$s_y$ (s/km)', labelpad = -1)
plt.title('G. Full Array', loc = 'left', fontsize = 'medium')


#%%
from matplotlib import cm
fig = plt.gcf()
fig.subplots_adjust(right=0.91)
cbar_ax = fig.add_axes([0.96, 0.08, 0.02, 0.45])
cbar = fig.colorbar(cm.ScalarMappable(norm=None, cmap=wyorm),cax=cbar_ax, label='Semblance (normalized)', ticks = [])
#%%
plt.gcf().set_size_inches(9, 9, forward=True) # max fig size for villarrica screen: 12.94x6.65 inch
plt.tight_layout()
#%%
plt.savefig('figures/FigS2_Array_response_3_6_22.png')
