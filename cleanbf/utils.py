import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd # needed for reading coords df as inv
import obspy

eps = 1e-12

def get_coordinates(x, y = None):
    """
    Finds sensor coordinates from various inputs
    
    Parameters:
    -----------
    x: either an array of x coordinates, obspy.Stream with trace.stats['coords'], or obspy.Inventory
    y: either an array of y coordinates (if x is an array of x coordinates), or None

    Returns:
    --------
    pandas.DataFrame with x, y, z, network, station, location fields. x and y are in km, z is in m.
    """
    if type(x) is obspy.Stream:
        try:
            ## Stream coordinates can either be lon/lat/z or x/y/z. x and y are km, z is m.
            ## This line will work if x has lon/lat/z coordinates, and will raise an exception
            ## if x has x/y/z coordinates.
            geometry = obspy.signal.array_analysis.get_geometry(x, coordsys = 'lonlat',
                                                              return_center = True)
            coords = {'x':geometry[:-1,0],
                      'y':geometry[:-1,1],
                      'z':geometry[:-1,2] + geometry[-1,2]} # last row in 'geometry' is coordinates of reference point
            # https://docs.obspy.org/packages/autogen/obspy.signal.array_analysis.get_geometry.html
        except:
            ## If we're here, then x is a stream with x/y/z coordinates. Extract them directly.
            ## We don't want to use get_geometry because it will pick a new center and shift the
            ## coordinates accordingly.
            coords = {'x': np.array([tr.stats.coordinates['x'] for tr in x]),
                      'y': np.array([tr.stats.coordinates['y'] for tr in x]),
                      'z': np.array([tr.stats.coordinates['elevation'] for tr in x])}

        coords['network'] = [tr.stats.network for tr in x]
        coords['station'] = [tr.stats.station for tr in x]
        coords['location'] = [tr.stats.location for tr in x]
    elif type(x) is obspy.Inventory:
        contents = x.get_contents()['channels']
        lats = [x.get_coordinates(s)['latitude'] for s in contents]
        lons = [x.get_coordinates(s)['longitude'] for s in contents]
        zz = [x.get_coordinates(s)['elevation'] for s in contents]
        xx = np.zeros(len(lats))
        yy = np.zeros(len(lats))
        for i, (lat, lon) in enumerate(zip(lats, lons)):
            xx[i], yy[i] = obspy.signal.util.util_geo_km(np.mean(lons), np.mean(lats), lon, lat)
        coords = {'x': xx, 'y': yy, 'z': zz,
                  'network': [string.split('.')[0] for string in contents],
                  'station': [string.split('.')[1] for string in contents],
                  'location': [string.split('.')[2] for string in contents]}
    else:
        coords = {'x':x,
                  'y':y,
                  'z':None,
                  'network':None,
                  'station':None,
                  'location':None}
    return pd.DataFrame(coords)

def _az_dist(a, b):
    """
    Find angular distance between two azimuths. Result is always between -180 and 180.
    """
    return ((a - b + 180) % 360) - 180

def _polar_transform(spec, sx_list, sy_list, az_start = 0, backazimuth = False):
    az_list = np.arange(36) * 10
    az_list = ((az_list - az_start) % 360) + az_start

    r_list = np.concatenate([sx_list[sx_list >= 0], [sx_list.max() + eps]])
    if len(spec.shape) == 3:
        polar_spec = np.zeros([spec.shape[0], len(az_list), len(r_list)])
    elif len(spec.shape) == 4:
        polar_spec = np.zeros([spec.shape[0], spec.shape[1], len(az_list), len(r_list)])
    for i, sx in enumerate(sx_list):
        for j, sy in enumerate(sy_list):
            if backazimuth:
                az = np.arctan2(-sx, -sy) * 180/np.pi
            else:
                az = np.arctan2(sx, sy) * 180/np.pi
            r = np.sqrt(sx**2 + sy**2)
            m = np.argmin(np.abs(_az_dist(az, az_list)))
            n = np.argmin(np.abs(r_list - r))
            if len(spec.shape) == 3:
                polar_spec[:,m,n] += spec[:,i,j]
            elif len(spec.shape) == 4:
                polar_spec[:,:,m,n] += spec[:,:,i,j]
    # drop the last r (unphysical excessive slowness)
    r_list = r_list[:-1]
    if len(spec.shape) == 3:
        polar_spec = polar_spec[:,:,:-1]
    elif len(spec.shape)==4:
        polar_spec = polar_spec[:,:,:,:-1]
       
    return (polar_spec, az_list, r_list)
