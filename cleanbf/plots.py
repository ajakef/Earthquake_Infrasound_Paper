import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd # needed for reading coords df as inv
import obspy
from cleanbf.utils import *
from cleanbf.utils import _polar_transform

eps = 1e-12

## white-yellow-orange-red colormap
wyor = LinearSegmentedColormap('wyor', {'red': [[0, 1, 1], 
                                                [1, 1, 1]],
                                        'green': [[0, 1, 1],
                                                  [0.075, 1, 1],
                                                  [0.25, 1, 1],
                                                  [0.925, 0, 0],
                                                  [1, 0, 0]],
                                        'blue': [[0, 1, 1],
                                                 [0.075, 1, 1],
                                                 [0.25, 0, 0],
                                                 [0.925, 0, 0],
                                                 [1, 0, 0]]
                                        })

wyorm = LinearSegmentedColormap('wyor', {'red': [[0, 1, 1], 
                                                [0.8, 1, 1],
                                                [1, 0.5, 0.5]],
                                        'green': [[0, 1, 1],
                                                  #[0.075, 1, 1],
                                                  [0.4, 1, 1],
                                                  [0.8, 0, 0],
                                                  [1, 0, 0]],
                                        'blue': [[0, 1, 1],
                                                 #[0.075, 1, 1],
                                                 [0.4, 0, 0],
                                                 [0.8, 0, 0],
                                                 [1, 0, 0]]
                                        })


def plot_freq_slow_spec(clean_output, plot_comp = 'fx', type = 'clean', semblance = True,
                        fRange = [0, np.Inf], sxRange = [-4, 4], 
                        syRange = [-4, 4], imageAdj = None, backazimuth = True,
                        ax = None):
    """
    Plot data from the 3-D frequency-slowness spectrum as a 2-D image.
    
    Parameters
    ----------
    clean_output: dict returned by clean()
    plot_comp: a string including one or two of 'f' (frequency), 'x' (x slowness), and 'y' (y 
         slowness). When two components are provided, they are the x and y axis, and power is 
         plotted as a color; when only one is provided, it is the x axis and power is the y axis.
    type: one of 'clean', 'original', or 'remaining'
    fRange: lower and upper limits for frequency axis
    sxRange: lower and upper limits for x slowness axis
    syRange: lower and upper limits for y slowness axis
    imageAdj: optional function to change scaling of power
    backazimuth: if True, flip the sx and sy axes to show the wave's direction of origin
    ax: optional; if given, axis to plot on. If not provided, the current figure/axis will be used,
         or if no axis is open, a new one will be created.
    
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    if type.lower() == 'clean':
        spec = clean_output['cleanSpec']
    elif type.lower() == 'original' or type.lower() == 'remaining':
        if semblance:
            total_power = np.real(np.einsum('iik->', clean_output[type.lower() + 'CrossSpec']))
            spec = clean_output[type.lower() + 'Spec'] / total_power
        else:
            spec = clean_output[type.lower() + 'Spec']
    else:
        raise "Invalid 'type' in plot_freq_slow_spec"

    sx = clean_output['sx']
    sy = clean_output['sy']
    f = clean_output['freq']
    if backazimuth:
        spec = np.flip(spec, [1,2]) # flip it over dimensions x and y, but not f
        sx = -np.flip(sx)
        sy = -np.flip(sy)
    wf = (f >= fRange[0]) & (f <= fRange[1])
    wsx = (sx >= sxRange[0]) & (sx <= sxRange[1])
    wsy = (sy >= syRange[0]) & (sy <= syRange[1])
    mat = np.einsum('fxy->' + plot_comp, spec[np.ix_(wf, wsx, wsy)]) # np.ix_ does multidimensional broadcasting
    if imageAdj is not None:
        mat = imageAdj(mat)
    indVars = {'f':f[wf], 'x': sx[wsx], 'y': sy[wsy]}
    if len(plot_comp) == 1:
        ax.plot(indVars[plot_comp], mat)
    if len(plot_comp) == 2:

        iv0 = indVars[plot_comp[0]]
        iv1 = indVars[plot_comp[1]]
        im = image(mat, iv0, iv1, zmin = 0, ax = ax)
        if 'f' in plot_comp:
            #aspect = 'auto'
            pass
        else:
            ax.axis('square')
        ax.set_xlabel(_comp_to_label(plot_comp[0], backazimuth))
        ax.set_ylabel(_comp_to_label(plot_comp[1], backazimuth))
    if (plot_comp == 'xy') or (plot_comp == 'yx'):
        _circle(1000/325, ax)
        _circle(1000/350, ax)
        _circle(1, ax) # seismic approximation
        if(type == 'original' or type == 'remaining'):
            #plt.clim([0,1])
            #cbar = plt.colorbar()
            cbar = fig.colorbar(im, ax = ax)
            cbar.set_label('Semblance')
            print('made colorbar')
    plt.title(type)
    return fig, ax
    #print('Remember plt.show()') # necessary in terminal

def polar_freq_slow_spec(clean_output, plot_comp = 'fh', type = 'clean', 
                         fRange = [-np.Inf, np.Inf], azRange = [-np.Inf, np.Inf], 
                         shRange = [0, 4], imageAdj = None, backazimuth = True,
                         ax = None):
    """
    Plot data from the 3-D frequency-slowness spectrum as a 2-D image.
    
    Parameters
    ----------
    clean_output: dict returned by clean()
    plot_comp: a string including one or two of 'f' (frequency), 'h' (horizontal slowness), and 'a' 
         (azimuth). When two components are provided, they are the x and y axis, and power is 
         plotted as a color; when only one is provided, it is the x axis and power is the y axis.
    type: one of 'clean', 'original', or 'remaining'
    fRange: lower and upper limits for frequency axis
    sxRange: lower and upper limits for x slowness axis
    syRange: lower and upper limits for y slowness axis
    imageAdj: optional function to change scaling of power
    backazimuth: if True, flip the sx and sy axes to show the wave's direction of origin
    ax: optional; if given, axis to plot on. If not provided, the current figure/axis will be used,
         or if no axis is open, a new one will be created.
    
    """
    if ax is None:
        ax = plt.gca()
    fig = ax.figure

    if type.lower() == 'clean':
        spec = clean_output['cleanSpec']
    elif type.lower() == 'original':
        spec = clean_output['originalSpec']
    elif type.lower() == 'remaining':
        spec = clean_output['remainingSpec']
    else:
        raise "Invalid 'type' in plot_freq_slow_spec"
    sx = clean_output['sx']
    sy = clean_output['sy']
    f = clean_output['freq']
    if backazimuth:
        spec = np.flip(spec, [1,2]) # flip it over dimensions x and y, but not f
        sx = -np.flip(sx)
        sy = -np.flip(sy)
    spec, az, sh = _polar_transform(spec, sx, sy)
    wf = (f >= fRange[0]) & (f <= fRange[1])
    waz = (az >= azRange[0]) & (az <= azRange[1])
    wsh = (sh >= shRange[0]) & (sh <= shRange[1])
    mat = np.einsum('fah->' + plot_comp, spec[np.ix_(wf, waz, wsh)]) # np.ix_ does multidimensional broadcasting
    if imageAdj is not None:
        mat = imageAdj(mat)
    indVars = {'f':f[wf], 'a': az[waz], 'h': sh[wsh]}
    if len(plot_comp) == 1:
        ax.plot(indVars[plot_comp], mat)
        ax.set_xlabel(_comp_to_label(plot_comp[0], backazimuth))
        ax.set_ylabel('Power')
    if len(plot_comp) == 2:
        iv0 = indVars[plot_comp[0]]
        iv1 = indVars[plot_comp[1]]
        image(mat, iv0, iv1, aspect = 'auto', zmin = 0)
        ax.set_xlabel(_comp_to_label(plot_comp[0], backazimuth))
        ax.set_ylabel(_comp_to_label(plot_comp[1], backazimuth))
    ax.set_title(type)
    #print('Remember plt.show()') # necessary in terminal



def _circle(r, ax = None):
    if ax is None:
        ax = plt.gca()
    az = np.arange(361) * np.pi/180
    ax.plot(r * np.cos(az), r * np.sin(az), 'k--', linewidth = 0.5)

def _comp_to_label(comp, backazimuth):
    if comp == 'f':
        return 'Frequency (Hz)'
    elif comp == 'x' and backazimuth:
        return 'x Back Slowness (s/km)'        
    elif comp == 'x' and not backazimuth:
        return 'x Forward Slowness (s/km)'
    elif comp == 'y' and backazimuth:
        return 'y Back Slowness'
    elif comp == 'y' and not backazimuth:
        return 'y Forward Slowness (s/km)'
    elif comp == 'a' and backazimuth:
        return 'Backazimuth (degrees)'
    elif comp == 'a' and not backazimuth:
        return 'Azimuth (degrees)'
    elif comp == 'h':
        return 'Horizontal Slowness (s/km)'
    else:
        return None
 
def plot_distances(st, limit, ha = 'center', va = 'bottom'):
    coords = get_coordinates(st)
    distance = calc_station_pair_distance(st)
    plt.plot(coords.x, coords.y, 'ko')
    nsta = coords.shape[0]
    for i in range(nsta):
        for j in range(i):
            if distance[i,j] < limit:
                plt.plot([coords.x[i], coords.x[j]], [coords.y[i], coords.y[j]], 'r-')
    for i in range(nsta):
        plt.text(coords.x[i], coords.y[i], st[i].stats.station + '.' + st[i].stats.location, va = va, ha = ha)

    plt.axis('square')


def calc_station_pair_distance(x, y = None):
    coords = get_coordinates(x, y)
    nsta = coords.shape[0]
    distance = np.zeros([nsta, nsta])
    for i in range(nsta):
        for j in range(nsta):
            distance[i,j] = np.sqrt((coords.x[i] - coords.x[j])**2 + (coords.y[i] - coords.y[j])**2)

    return distance


def make_station_mask(stream, exclude = []):
    nsta = len(stream)
    mask = np.ones([nsta, nsta])
    for i, tr in enumerate(stream):
        if tr.stats.station in exclude:
            mask[i,:] = 0
            mask[:,i] = 0
    return mask



def image(Z, x = None, y = None, aspect = 'equal', zmin = None, zmax = None, ax = plt, crosshairs=True):
    # Z rows are x, columns are y
    if x is None:
        x = np.arange(Z.shape[0])
    if y is None:
        y = np.arange(Z.shape[1])
    #im = ax.imshow(Z.transpose(), extent = [x[0], x[-1], y[0], y[-1]], aspect = aspect, 
    #           origin = 'lower', vmin = zmin, vmax = zmax, cmap = 'YlOrRd')
    im = ax.pcolormesh(x, y, Z.T, vmin = zmin, vmax = zmax, cmap=wyor, shading = 'auto')
    if crosshairs:
        ax.hlines(0, x[0], x[-1], 'k', linewidth=0.5)
        ax.vlines(0, y[0], y[-1], 'k', linewidth=0.5)
    return im



