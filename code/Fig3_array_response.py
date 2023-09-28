import obspy
import obspy.signal.array_analysis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

import pickle
import cleanbf
from cleanbf import wyorm

def slowness_image(M, lim, zlim = [0, 1]):
    plt.imshow(M.transpose(), extent = [-lim, lim, -lim, lim], origin='lower', vmin=zlim[0], vmax = zlim[1], cmap = wyorm)
    return plt.colorbar()

def circle(r):
    # run plt.plot(*circle(r))
    theta = np.arange(361) * np.pi/180
    return (r*np.sin(theta), r*np.cos(theta))

sub_array = np.array([17, 9, 14])-1
gs = gridspec.GridSpec(7, 2)

## clean parameters to use
s_list = np.arange(-4, 4, 0.1)

inv = obspy.read_inventory('data/XP_PARK_inventory.xml') # includes coordinates
coords_full = cleanbf.get_coordinates(inv)
coords_sub = coords_full.iloc[sub_array,:]
#%% A: map, with 9, 14, and 17 marked
plt.subplot(gs[:3,:])
plt.scatter(1000*coords_sub.x, 1000*coords_sub.y, facecolor = 'white', edgecolor = 'red', marker = '^', s = 70)
plt.scatter(1000*coords_full.x, 1000*coords_full.y, color = 'black', marker = 'o', s = 10)
for i in range(22):
    plt.text(1000*coords_full.x[i]+2, 1000*coords_full.y[i], coords_full.location[i], )

plt.axis('equal')
plt.xlabel('Easting (m)')
plt.ylabel('Northing (m)')
plt.title('a. Station Map for XP.PARK', loc='left')
plt.axis('equal')
plt.legend(['Sub-array (N=3)', 'Full Array (N=22)'], loc = 'upper right', fontsize = 'small')

#%% B: 3-element array response (wavenumber)
plt.subplot(gs[3:5,0])
xy = np.append(np.array(coords_sub.loc[:,['x', 'y']]),np.zeros([3,1]), axis=1)

klim = 2*np.pi*10/0.330 # wavenumber at 10 Hz in rad/km
kstep = 1

arrayResp = obspy.signal.array_analysis.array_transff_wavenumber(xy, coordsys = 'xy', klim=klim, kstep=kstep)

slowness_image(arrayResp, klim/100)

plt.plot(*circle(2*np.pi * 2 / 0.333/100), 'k--')
plt.plot(*circle(2*np.pi * 10 / 0.333/100), 'k--')
plt.title('b. Array Response (N=3)', loc = 'left')
plt.xlabel('$k_x$ (×$10^2$ rad/km)', labelpad = -1)
plt.ylabel('$k_y$ (×$10^2$ rad/km)', labelpad = -1)


#%% C: PARK response (wavenumber)
plt.subplot(gs[3:5,1])
xy = np.append(np.array(coords_full.loc[:,['x', 'y']]), np.zeros([22,1]), axis=1)

klim = 2*np.pi*10/0.330 # wavenumber at 10 Hz in rad/km
kstep = 1

arrayResp = obspy.signal.array_analysis.array_transff_wavenumber(xy, coordsys='xy', klim=klim, kstep=kstep)

cbar = slowness_image(arrayResp, klim/100)
cbar.set_label('Semblance')
plt.plot(*circle(2*np.pi * 2 / 0.333/100), 'k--')
plt.plot(*circle(2*np.pi * 10 / 0.333/100), 'k--')
plt.title('c. Array Response (N=22)', loc = 'left')
plt.xlabel('$k_x$ (×$10^2$ rad/km)', labelpad = -1)
plt.ylabel('$k_y$ (×$10^2$ rad/km)', labelpad = -1)

#%% D: 3-element slowness spectrum
t_index = 14 # 14: multiple sonic waves. 9: seismic wave with aliasing.
plt.subplot(gs[5:,0])
with open('data/pkl/clean_aftershock_3.pkl', 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
    o_sub = output['original'][t_index,:,:]
    o_sub = o_sub/o_sub.max() * output['original_semblance'][t_index]

slowness_image(o_sub, s_list.max(), [o_sub.min(), o_sub.max()])
plt.plot(*circle(3), 'k--')
plt.xlabel('$s_x$ (s/km)', labelpad = -1)
plt.ylabel('$s_y$ (s/km)', labelpad = -1)
plt.title('d. Slowness Spectrum (N=3)', loc = 'left')
#%% E: PARK slowness spectrum
plt.subplot(gs[5:7,1])
with open('data/pkl/clean_aftershock_full.pkl', 'rb') as file:
    d = pickle.load(file)
    locals().update(d)
    o_full = output['original'][t_index,:,:]
    o_full = o_full/o_full.max() * output['original_semblance'][t_index]

cbar = slowness_image(o_full, s_list.max(), [o_full.min(), o_full.max()])
cbar.set_label('Semblance')
plt.plot(*circle(3), 'k--')

plt.xlabel('$s_x$ (s/km)', labelpad = -1)
plt.ylabel("""$s_y$ (s/km)""", labelpad = -1)
plt.title('e. Slowness Spectrum (N=22)', loc = 'left')




#%%
plt.gcf().set_size_inches(6.5, 9, forward=True) # max fig size for villarrica screen: 12.94x6.65 inch
plt.tight_layout()
#%%
plt.savefig('figures/Fig3_array_response.png', dpi=300)

####################
