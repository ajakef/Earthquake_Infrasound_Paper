import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd # needed for reading coords df as inv
import obspy

def make_synth_stream(Nt = 400, dt = 0.01, x = None, y = None, dx = None, dy = None, Nx = None, 
                    Ny = None, sx = None, sy = None, amp = None, fl = None, fh = None, fc = None, 
                    uncorrelatedNoiseAmp = 0, prewhiten = True):
    r"""
    Make synthetic array data including any number of waves with defined slowness, amplitude, and bandwidth. Each wave consists of filtered white noise.
    
    :type Nt: int
    :param Nt: output trace length (samples)
    :type dt: float
    :param dt: sample interval (seconds)
    :type x: numpy.array
    :param x: x coordinates of sensors in array (km)
    :type y: numpy.array
    :param y: y coordinates of sensors in array (km)
    :type dx: float
    :param dx: spacing between sensors in x-direction if x is not provided (km)
    :type dy: float
    :param dy: spacing between sensors in y-direction if y is not provided (km)
    :type Nx: int
    :param Nx: number of sensors in x direction if x is not provided (km)
    :type Ny: int
    :param Ny: number of sensors in y direction if y is not provided (km)
    :type sx: numpy.array
    :param sx: wave slownesses in x direction (s/km)
    :type sy: numpy.array
    :param sy: wave slownesses in y direction (s/km)
    :type amp: numpy.array
    :param amp: wave amplitudes
    :type fl: numpy.array
    :param fl: low corner frequencies of waves (Hz)
    :type fh: numpy.array
    :param fh: high corner frequencies of waves (Hz). If fh is too close to fl, instability can occur
    :type fc: numpy.array
    :param fc: if fl and fh are not provided, the logarithmic "center" of each wave's 2-octave passband (Hz)
    :type uncorrelatedNoiseAmp:
    :param uncorrelatedNoiseAmp: amplitude of uncorrelated noise added to each trace
    :return: :class:`obspy.stream`
    """
    x, y = _make_default_coords(x, y, dx, dy, Nx, Ny)
    sx, sy, amp, fl, fh = _make_default_waves(sx, sy, amp, fl, fh, fc)
    Nwaves = sx.size
    t = np.arange(Nt) * dt
    overSample=10; Ntt = 2*overSample*Nt; dtt = dt/overSample
    tt = (np.arange(Ntt)-Ntt/3)*dtt # provides padding and higher time res for interp (e.g., 0-1 s becomes -0.5 - 1.5 s)
    stf_list = []
    for i in range(Nwaves):
        [b,a]=scipy.signal.butter(4, [2*fl[i]*dtt,2*fh[i]*dtt], 'bandpass')
        signal = scipy.signal.lfilter(b, a, _make_noise(Ntt, prewhiten))
        signal *= amp[i]/np.std(signal)
        stf_list.append(scipy.interpolate.interp1d(tt, signal, fill_value='extrapolate', kind='cubic'))
    stream = obspy.Stream()
    for i in range(len(x)):
        data = _make_noise(Nt, prewhiten)
        data *= uncorrelatedNoiseAmp / np.std(data)
        for j in range(Nwaves):
            data += stf_list[j](t - x[i] * sx[j] - y[i] * sy[j])
        tr = obspy.Trace(data)
        loc = obspy.core.AttribDict()
        loc['x'] = x[i]
        loc['y'] = y[i]
        loc['elevation'] = 0
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
        tr.stats.delta = dt
        tr.stats.location = str(i)
        stream += tr
    return stream

def make_synth_stream_local(Nt = 400, dt = 0.01, x = None, y = None, dx = None, dy = None, Nx = None, 
                    Ny = None, src_xyz = None, amp = None, fl = None, fh = None, fc = None, 
                    uncorrelatedNoiseAmp = 0, prewhiten = True, c = 0.340):
    x, y = _make_default_coords(x, y, dx, dy, Nx, Ny)
    #sx, sy, amp, fl, fh = _make_default_waves(sx, sy, amp, fl, fh, fc)
    nsrc = src_xyz.shape[0]
    nsta = len(x)
    sta_xyz = np.zeros([len(x), 3])
    sta_xyz[:,0] = x; sta_xyz[:,1] = y
    t = np.arange(Nt) * dt
    overSample=10; Ntt = 2*overSample*Nt; dtt = dt/overSample
    tt = (np.arange(Ntt)-Ntt/3)*dtt # provides padding and higher time res for interp (e.g., 0-1 s becomes -0.5 - 1.5 s)
    stf_list = []
    for i in range(nsrc):
        [b,a]=scipy.signal.butter(4, [2*fl[i]*dtt,2*fh[i]*dtt], 'bandpass')
        signal = scipy.signal.lfilter(b, a, _make_noise(Ntt, prewhiten))
        signal *= amp[i]/np.std(signal)
        stf_list.append(scipy.interpolate.interp1d(tt, signal, fill_value='extrapolate', kind='cubic'))
    stream = obspy.Stream()
    for i in range(nsta):
        data = _make_noise(Nt, prewhiten)
        data *= uncorrelatedNoiseAmp / np.std(data)
        for j in range(nsrc):
            distance = np.sqrt(np.sum((src_xyz[j,:] - sta_xyz[i,:])**2)) + 1e-6 # epsilon to prevent zero division
            data += stf_list[j](t - distance/c) / distance
        tr = obspy.Trace(data)
        loc = obspy.core.AttribDict()
        loc['x'] = x[i]
        loc['y'] = y[i]
        loc['elevation'] = 0
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
        tr.stats.delta = dt
        tr.stats.location = str(i)
        stream += tr
    return stream

def _make_noise(N, prewhiten = True):
    x = np.random.normal(0,1,N)
    if prewhiten:
        s = np.fft.fft(x)
        x = np.real(np.fft.ifft(s/(1e-50 + np.abs(s)))) # water level to prevent div-by-0
    return x


def _assign_default_none(var, default):
    if var is None or np.array(var).size == 0:
        return np.array(default)
    else:
        return np.array(var)

def _make_default_waves(sx = None, sy = None, amp = None, fl = None, fh = None, fc = None):
    assert ((fl is None) and (fh is None)) or \
            ((fl is not None) and (fh is not None) and (len(np.array(fl)) == len(np.array(fh)))), \
            'Check fl and fh'
    
    sx = _assign_default_none(sx, default = 0)
    sy = _assign_default_none(sy, default = 0)
    amp = _assign_default_none(amp, default = 1)
    fc = _assign_default_none(fc, default = 3)
    fl = _assign_default_none(fl, default = fc / 2)
    fh = _assign_default_none(fh, default = fc * 2)

    ## make all the outputs the same length
    maxLength = max([var.size for var in (sx, sy, amp, fc, fl, fh)])
    assert (sx.size == maxLength) or (sx.size == 1)
    if sx.size == 1:
        sx = sx + np.zeros(maxLength)
    assert (sy.size == maxLength) or (sy.size == 1)
    if sy.size == 1:
        sy = sy + np.zeros(maxLength)
    assert (amp.size == maxLength) or (amp.size == 1)
    if amp.size == 1:
        amp = amp + np.zeros(maxLength)
    assert (fl.size == maxLength) or (fl.size == 1)
    if fl.size == 1:
        fl = fl + np.zeros(maxLength)
    assert (fh.size == maxLength) or (fh.size == 1)
    if fh.size == 1:
        fh = fh + np.zeros(maxLength)
    return sx, sy, amp, fl, fh

def _make_default_coords(x=None, y=None, dx=None, dy=None, Nx=None, Ny=None):
    defaultDiff = 20/1000
    if (x is not None) and (y is not None):
        assert len(x) == len(y)
    elif (x is not None):
        x = np.array(x)
        y = 0 * x
    elif (y is not None):
        y = np.array(y)
        x = 0 * y
    else: # both x and y are None; create x and y as a grid
        if dx is None:
            dx = defaultDiff
        if dy is None:
            dy = defaultDiff
        if (Nx is not None) and (Nx > 1) and (Ny is not None) and (Ny > 1):
             xtmp = (np.arange(Nx) - (Nx-1)/2) * dx
             ytmp = (np.arange(Ny) - (Ny-1)/2) * dy
             x, y = np.meshgrid(xtmp, ytmp) # needs to be reshaped; see last line
        elif (Nx is not None) and (Nx > 1):
             x = (np.arange(Nx) - (Nx-1)/2) * dx
             y = 0 * x
        elif (Ny is not None) and (Ny > 1):
             y = (np.arange(Ny) - (Ny-1)/2) * dy
             x = 0 * y
        else:
            Nx = 3
            x = (np.arange(Nx) - (Nx-1)/2) * dx
            y = 0 * x
    return np.array(x).reshape(x.size), np.array(y).reshape(y.size)

