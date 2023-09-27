import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import pandas as pd # needed for reading coords df as inv
import obspy
from obspy.signal.invsim import cosine_taper
from obspy.signal.util import next_pow_2
import obspy.signal.array_analysis

from cleanbf.utils import _polar_transform, get_coordinates
from cleanbf.plots import image

#try:
#    # mtspec can be used to calculate spectra and cross-spectra using multitaper method,
#    # but it's still experimental and not recommended.
#    import mtspec
#except:
#    pass
#    #print("Library mtspec is not available, meaning that multitaper spectra are not an option")

eps = 1e-12


def make_steering_vectors(stream, freqs, xSlowness, ySlowness = [0]):
    # plane wave steering vector dimensions are 
    # nf, nsx, nsy, nsta
    # exp(1j * 2 * pi * f * dt)
    geometry_array = get_coordinates(stream)
    if isinstance(geometry_array, list) or isinstance(geometry_array, pd.DataFrame):
        geometry_array = np.array(geometry_array.loc[:,['x', 'y']])
    fs = stream[0].stats.sampling_rate
    #deltaf = freqs[1] - freqs[0]

    nf = len(freqs)
    nsta = geometry_array.shape[0]
    nsx = len(xSlowness)
    nsy = len(ySlowness)

    ff, sx, sy, array_xx = np.meshgrid(freqs, xSlowness, ySlowness, geometry_array[:,0], indexing = 'ij')
    ff, sx, sy, array_yy = np.meshgrid(freqs, xSlowness, ySlowness, geometry_array[:,1], indexing = 'ij')
    steering_vectors = np.exp(-2j * np.pi * ff * (sx*array_xx + sy*array_yy))
    return steering_vectors
    
def make_steering_vectors_local(stream, geometry_src, freqs, c = 0.340):
    # plane wave steering vector dimensions are 
    # nf, nsx, nsy, nsta
    # exp(1j * 2 * pi * f * dt)
    geometry_array = get_coordinates(stream)
    if isinstance(geometry_array, list):
        geometry_array = np.array(geometry_array).transpose()
    fs = stream[0].stats.sampling_rate
    nf = len(freqs)
    nsta = geometry_array.shape[0]
    nsrc = geometry_src.shape[0]
    ## runtime difference of 10-100x when vectorized: result is shape (nf, nsrc, 1, nsta) including placeholder for compatibility
    ff, src_xx, pp, array_xx = np.meshgrid(freqs, geometry_src[:,0], [0], geometry_array.x, indexing = 'ij')
    ff, src_yy, pp, array_yy = np.meshgrid(freqs, geometry_src[:,1], [0], geometry_array.y, indexing = 'ij')
    dd = np.sqrt((array_xx-src_xx)**2 + (array_yy-src_yy)**2)
    steering_vectors = np.exp(-2j * np.pi * ff * dd/c) / dd
    return steering_vectors

def calc_fourier_window(stream, offset=0, taper = None, taper_param = 4, raw=False, prewhiten = False):
    spoint = np.zeros(len(stream))
    nsamp = len(stream[0])
    nfft = next_pow_2(nsamp)
    nlow = 1
    nhigh = nfft // 2 - 1
    nf = nhigh - nlow + 1  # include upper and lower frequency
    freqList = np.arange(nlow, nlow+nf)/(nfft * stream[0].stats.delta)
    if (taper is None) or (taper.lower() == 'tukey'):
        taper = cosine_taper(nsamp, p=0.22)
    ft = np.empty((len(stream), nf), dtype=np.complex128)
    for i, tr in enumerate(stream):
        dat = tr.data[int(spoint[i] + offset):int(spoint[i] + offset + nsamp)]
        if False:#(type(taper) is str) and (taper.lower() == 'multitaper'):
            #ft[i, :] = mtspec(dat, stream[0].stats.delta, time_bandwidth = taper_param, nfft=nfft)[nlow:nlow + nf]
            pass
        else:
            if not raw:
                dat = scipy.signal.detrend(dat) * taper
            ft[i, :] = np.fft.rfft(dat, nfft)[nlow:nlow + nf]
    ft *= np.sqrt(2*stream[0].stats.delta / nsamp) # one-sided spectrum with physical amplitude units
    if prewhiten:
        eps = np.min(np.abs(ft)[ft != 0])
        ft /= np.abs(ft) + eps # apply a tiny waterlevel to avoid divide-by-zero
    return ft, freqList

def calc_cross_spectrum(stream, offset=0, taper=None, taper_param = 4, raw=False, win_length_sec = 1, 
                        freq_bin_width = 1, prewhiten = False, plot = False, criterion_function = None):
    """
    Calculates the cross-spectrum of traces in an obspy stream.
    
    Parameters
    ----------
    stream: obspy stream containing array data (and optionally coordinates in AttribDicts)
    taper: used to taper time windows before calculating spectrum. Can be either default 'Tukey' or 
         'multitaper', or a numpy.array containing the user-defined taper.
    taper_param: unused for Tukey; refers to the time-bandwidth product for multitaper.
    raw: if True, do not apply tapers or detrending when calculating spectrum and cross-spectrum
    freq_bin_width: number of adjacent samples in spectrum contained in each frequency bin
    prewhiten: if True, prewhiten spectra before calculating cross-spectrum
    plot: if True, sum the cross-spectrum's modulus over frequency and plot as image
    criterion_function: function of (stream) that returns a square boolean np.array,
         determining whether a time window is included in the cross-spectrum calculation.
    
    Returns
    -------
    cross_spec_binned: binned cross-spectrum (indices trace 1, trace 2, freq)
    FT: list of individual trace spectra
    freqs_binned: list of frequencies corresponding to frequency indices
    dfn: numerator degrees of freedom 
    dfd: denominator degrees of freedom
    """
    if criterion_function is None:
        def criterion_function(st): return np.ones([len(st), len(st)])
    cross_spec = 0
    FT = 0 
    if False:#(taper is not None) and (taper.lower() == 'multitaper'):
        #num_windows = 1
        ##n_freq = int(2**np.ceil(np.log2(len(stream[0].data)))/2)
        #n_freq = int(len(stream[0].data)/2)
        #cross_spec = np.zeros([len(stream), len(stream), n_freq], dtype = 'complex')
        #for i in range(len(stream)):
        #    for j in range(i):
        #        mt_output = mtspec.mt_coherence(stream[0].stats.delta, stream[i].data, 
        #                                        stream[j].data, tbp = taper_param, kspec = 5, 
        #                                        nf = n_freq, p = 0.95, cohe = True, phase = True, 
        #                                        speci = True, specj = True, freq = True)
        #        df = np.diff(mt_output['freq'])[0]
        #        ## Hack to enforce Parseval's relation. mtspec doesn't do it.
        #        scale_i = np.var(stream[i])/(np.sum(mt_output['speci']) * df)
        #        scale_j = np.var(stream[j])/(np.sum(mt_output['specj']) * df)
        #        cross_spec[i,i,:] = mt_output['speci'] * scale_i
        #        cross_spec[j,j,:] = mt_output['specj'] * scale_j
        #        cross_spec[i,j,:] = (mt_output['speci'] * mt_output['specj'] * scale_i * scale_j * 
        #                            mt_output['cohe'])**0.5 * np.exp(1j * np.pi/180 * mt_output['phase'])
        #        cross_spec[j,i,:] = cross_spec[i,j].conj()
        #freqs = mt_output['freq']
        pass
    else: # use default Tukey taper
        win_length_samp = int(np.round(win_length_sec / stream[0].stats.delta))
        data_length_samp = len(stream[0].data)
        overlap = 0.5 # for possible future use. calculating degrees of freedom assumes non-overlapping windows.
        num_windows = 1 + int(np.ceil((data_length_samp - win_length_samp) / (win_length_samp * (1 - overlap))))
        if num_windows < 1:
            raise Exception('No analysis windows for calculating cross spectrum...empty stream?')
        criterion = np.zeros([len(stream), len(stream)])
        criterion_count = np.zeros([len(stream), len(stream)])
        
        for i in range(num_windows):
            win_start = int(np.round(i * (data_length_samp - win_length_samp) / (num_windows-1 + eps)))
            tmpStream = stream.copy()
            for tr in tmpStream:
                tr.data = tr.data[win_start:(win_start + win_length_samp)]
            criterion = criterion_function(tmpStream)
            criterion_count += criterion
            tmpFT, freqs = calc_fourier_window(tmpStream, offset, taper, taper_param, raw, prewhiten)
            cross_spec += np.einsum('ik,jk,ij->ijk', tmpFT, tmpFT.conj(), criterion) # identical to _r from obspy, but reordered indices. i: station, j: station, k: freq
            FT += np.abs(tmpFT) / num_windows
        cross_spec = np.einsum('ijk,ij->ijk', cross_spec, 1/criterion_count)
    ## bin smoothing
    n_bins = len(freqs) // freq_bin_width
    cross_spec_binned = np.zeros((len(stream), len(stream), n_bins), dtype = np.complex128)
    freqs_binned = np.zeros(n_bins)
    for i in range(n_bins):
        cross_spec_binned[:,:,i] += np.mean(cross_spec[:,:,(i*freq_bin_width):((i+1)*freq_bin_width)], axis = 2)
        freqs_binned[i] = np.mean(freqs[(i*freq_bin_width):((i+1)*freq_bin_width)])
    
    ## calculate degrees of freedom
    dfn = 2 * (freq_bin_width-1 + num_windows) # so that bin width 1, having no extra freqs, adds no extra df
    dfd = (len(stream)-1) * dfn
    if plot:
        image(np.einsum('ijk->ij', np.abs(cross_spec_binned)))
    return cross_spec_binned, FT, freqs_binned, dfn, dfd



####################################################
####################################################
####################################################
####################################################
####################################################
def calc_weights(steering_vectors):
    # denominator: RSS over stations for each freq and source. 
    weights_denom = np.sqrt(np.einsum('ijkl,ijkl->ijk', (steering_vectors**1).conj(), steering_vectors**1))
    weights = steering_vectors.copy()#.conj() # freq, sx, sy, station
    for i in range(weights.shape[3]):
        weights[:,:,:,i] /= weights_denom    
    ## now, sum of squares over stations is 1 for every freq-sx-sy combo
    return weights
 
def calc_freq_slowness_spec(cross_spec, steering_vectors = None, weights = None, freq_to_use = None, mask = None):
    if weights is None:
        weights = calc_weights(steering_vectors)
    if freq_to_use is None:
        freq_to_use = np.ones(cross_spec.shape[2])
    if mask is None:
        mask = np.ones(cross_spec.shape[:2])
    return np.abs(np.einsum('k,klmi,ijk,klmj,ij->klm', freq_to_use, weights.conj(), cross_spec, weights, mask)) # n_freq x n_slow x n_slow; absolute power over f, sx,sy

def clean_step(cleanSpec, cross_spec, wB, phi, stopF, separate_freqs, verbose, sxList, syList, show_plots, cross_spec_mask):
    originalCrossSpec = cross_spec.copy()
    dropped_power = 0 * cross_spec
    nsta = cross_spec.shape[0]
    done = False
    positive_freqs = np.ones(cross_spec.shape[2]) # these get set to zero when a frequency's power starts to get negative
    count = 0
    while not done:
        count += 1
        ## calculate frequency-slowness spectrum
        P = calc_freq_slowness_spec(cross_spec, weights = wB, freq_to_use = positive_freqs, mask = cross_spec_mask)
        if separate_freqs == 1:  ## use this code to pick best slowness AND best frequency
            f_inds,imax,jmax= np.where(P == P.max()); f_inds = f_inds[:1]; imax=imax[0]; jmax=jmax[0]; #kmax=kmax[0]
            f_inds = f_inds == np.arange(len(positive_freqs)) ## convert f_inds to bool
            Pmax = P[f_inds,imax,jmax]
            Ptotal = (np.einsum('iik->k', np.abs(cross_spec)) * positive_freqs)[f_inds]
        elif separate_freqs == 2: ## try to be smart about which frequencies to work with
            PP = np.abs(np.einsum('ijk->jk', P))
            Pmax = PP.max()
            imax,jmax = np.where(PP == Pmax); imax=imax[0]; jmax=jmax[0]
            ## pick freqs that, at imax & jmax, are 1) near their max power and 2) significant
            #breakpoint()
            f_inds = (P[:,imax,jmax] > np.einsum('iik->k', np.abs(cross_spec)) / 
                               (1 + (nsta-1)/stopF)) & (P[:,imax,jmax] > (0.99*np.amax(P, (1,2))))
            Ptotal = np.einsum('iik->', np.abs(cross_spec[:,:,f_inds]))
            Pmax = np.sum(P[f_inds, imax, jmax])
        else: ## pick best slowness over all frequencies
            PP = np.abs(np.einsum('ijk->jk', P))
            Pmax = PP.max()
            imax,jmax = np.where(PP == Pmax); imax=imax[0]; jmax=jmax[0]
            #f_inds = cross_spec[0,0,:]*0 > -1 # throwaway to mean all f indices
            f_inds = (positive_freqs == 1)
            Ptotal = np.einsum('iik->', np.abs(cross_spec[:,:,f_inds]))
            Pmax = np.sum(P[f_inds, imax, jmax])

        ## Test whether the optimum slowness is coherent enough. If yes, adjust outputs accordingly. If not, break.
        stopCondition = Ptotal/(1 + (nsta-1)/stopF)
        remainingPower = np.abs(np.einsum('iij->', cross_spec) / np.einsum('iij->', originalCrossSpec))
        if (Pmax > stopCondition) and (remainingPower > 0.001):
            done = False
            compToRemove = phi * np.einsum('i,ij,ik->jki',P[:,imax,jmax], wB[:,imax,jmax,:],
                                     wB[:,imax,jmax,:].conj())
            for i in np.where(f_inds)[0]:
                #if any([any(newCrossSpec[i,i,:] < 0) for i in range(nsta)]):
                #if any(np.diag(cross_spec[:,:,i] - compToRemove[:,:,i]) < 0):
                diags = np.diag(cross_spec[:,:,i] - compToRemove[:,:,i])
                if np.mean(diags) < np.std(diags):# allow some negativity, but not much
                    if verbose: print('Dropping freq index %d due to negative power' %i) 
                    positive_freqs[i] = 0 
                    dropped_power[:,:,i] = cross_spec[:,:,i]
                    
            # this section causes problems with separate_freqs = 1...comment for now                        
            #if np.sum(compToRemove[:,:,f_inds & (positive_freqs == 1)]) == 0:
            #    breakpoint()

            ## if remaining power is > 1, there's a problem
            #if remainingPower > (1+1e-9):
            #    breakpoint()
            cleanSpec[f_inds, imax, jmax] += phi * P[f_inds, imax, jmax] * positive_freqs[f_inds]
            if show_plots:
                plot_slowness_spectrum(cross_spec, compToRemove, cleanSpec, originalCrossSpec, wB, sxList, syList, count)
                
            cross_spec[:,:,f_inds] -= np.einsum('ij,ijk,k->ijk', cross_spec_mask, compToRemove, positive_freqs)[:,:,f_inds]
            #cross_spec = DropNegatives(cross_spec) ## shouldn't run anymore but sends an alert if it does
            if verbose: print('sx %.3f, sy %.3f, p/stop %.3f, remaining %.3f' % (sxList[imax], syList[jmax], Pmax/stopCondition, remainingPower))
        else:
            done = True

    return {'cleanSpec':cleanSpec, 'cross_spec':cross_spec, 'remainingSpec':P, 'droppedCrossSpec':dropped_power}

def plot_slowness_spectrum(cross_spec, compToRemove, cleanSpec, originalCrossSpec, wB, sxList, syList, count):
    """
    Used to plot snapshot slowness spectra showing clean power and remaining power at successive 
    clean iterations, using the "show_plots" option in clean().
    """
    fig, ax = plt.subplots(3,1,squeeze=False)
    fig.set_tight_layout(True)
    ref_power = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), originalCrossSpec, wB)).max()
    P = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), cross_spec, wB)) # n_slow x n_slow; absolute power over sx,sy
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[0,0])
    ax[0,0].set_xlabel('x slowness (s/km)')
    ax[0,0].set_ylabel('y slowness (s/km)')
    ax[0,0].set_title('Slowness Spectrum')
    cb = fig.colorbar(im, ax=ax[0,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')    
    P = np.abs(np.einsum('klmi,ijk,klmj->lm', wB.conj(), compToRemove, wB)) # n_slow x n_slow; absolute power over sx,sy
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[1,0])
    ax[1,0].set_xlabel('x slowness (s/km)')
    ax[1,0].set_ylabel('y slowness (s/km)')
    ax[1,0].set_title('Component to Remove')
    cb = fig.colorbar(im, ax=ax[1,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')
    P = np.abs(np.einsum('ijk->jk', cleanSpec)) # f, sx, sy; sum over f
    im = image(P, sxList, syList, zmin = 0, zmax = ref_power, ax = ax[2,0])
    ax[2,0].set_xlabel('x slowness (s/km)')
    ax[2,0].set_ylabel('y slowness (s/km)')
    ax[2,0].set_title('New Clean Spectrum')
    cb = fig.colorbar(im, ax=ax[2,0])
    cb.ax.set_ylabel('Power ($Pa^2$)')
    fig.set_figheight(10)
    fig.set_figwidth(4)
    fig.tight_layout()
    fig.savefig('plot_step_%.0f.png' % count)
    plt.close(fig)
    if count > 3: raise Exception()
#####################################################
## Function to step through a stream and clean all time windows
def clean_loop(stream, loop_step = 1, loop_width=2, x = None, y = None, sxList = None, syList = None, phi = 0.1, p_value = 0.01, 
          win_length_sec = 1, freq_bin_width = 1, freq_min = 0, freq_max = 15, separate_freqs = 0,
          rawFT = False, taper = 'Tukey', taper_param = 4, prewhiten = False,
          show_plots = False, verbose = True):
    t1 = loop_start = stream[0].stats.starttime
    loop_end = stream[0].stats.endtime
    nt = int(1 + (loop_end - loop_start - loop_width) // loop_step)
    z = np.zeros(nt)

    dirty_output = {'t':z, 'sx':z, 'sy':z, 'power':z}
    
    for i in range(nt):
        st = stream.slice(t1, t1 + loop_width)
        halftime = t1 + loop_width/2 - loop_start
        print('%f of %f' % (halftime, loop_end - loop_start))
        result = clean(st, verbose = verbose, phi = phi, separate_freqs = separate_freqs, win_length_sec = win_length_sec,
                                  freq_bin_width = freq_bin_width, freq_min = freq_min, freq_max = freq_max, 
                                  sxList = sxList, syList = syList, prewhiten = False)
        if t1 == loop_start: # allocate the output variables
            freq = result['freq']
            output = {'t':np.zeros(nt), 'sx':sxList, 'sy':syList, 'f':freq, 
                      'original_power':np.zeros(nt), 'original_semblance':np.zeros(nt),
                      'original_sx':np.zeros(nt), 'original_sy':np.zeros(nt), 
                      'clean':np.zeros((nt, len(freq), len(sxList), len(syList))), 
                      'original':np.zeros((nt, len(sxList), len(syList))),
                      'all_data':[]}
        ## populate the output variables
        output['t'][i] = halftime
        output['all_data'] = result
        output['clean'][i,:,:,:] = result['cleanSpec']
        output['original'][i,:,:] = np.einsum('ijk->jk',result['originalSpec'])
        j_sx, k_sy = np.where(output['original'][i,:,:] == output['original'][i,:,:].max())
        output['original_power'][i] = output['original'][i,j_sx,k_sy]
        output['original_semblance'][i] = calc_semblance(result).max()
        output['original_sx'][i] = sxList[j_sx]
        output['original_sy'][i] = sxList[k_sy]
        t1 += loop_step
        
 
    output['original_sh'] = np.sqrt(output['original_sx']**2+output['original_sy']**2)
    output['original_az'] = np.arctan2(output['original_sx'], output['original_sy']) * 180/np.pi
    output['clean_polar'], output['az'], output['sh'] = _polar_transform(output['clean'], sxList, syList)
    output['clean_polar_back'], output['back_az'], output['sh'] = _polar_transform(output['clean'], sxList, syList, backazimuth = True)

    return output    
#####################################################
## Function to iteratively remove wavefield components from obspy stream
def clean(stream, x = None, y = None, sxList = None, syList = None, phi = 0.1, p_value = 0.01, 
          win_length_sec = 1, freq_bin_width = 1, freq_min = 0, freq_max = 15, separate_freqs = 0,
          rawFT = False, taper = 'Tukey', taper_param = 4, prewhiten = False, 
          show_plots = False, verbose = True, steering_vectors = None, cross_spec = None, freqList = None,
          cross_spec_mask = None, criterion_function = None):
    """
    Iteratively deconvolves array response from cross-spectrum, stopping when the F statistic drops
    below a threshold and resulting in a sparse 3-D array (frequency, x-slowness, y-slowness).

    Parameters
    ----------
    stream: obspy stream containing array data (and optionally coordinates in AttribDicts)
    x, y: coordinates of array elements, if not contained in stream (km, for consistency with obspy)
    sxList, syList: lists of x slownesses and y slownesses (sec/km, for consistency with obspy)
    phi: in each step, the proportion of energy at the optimal slowness removed from the 
         cross-spectrum. Low phi means more precision but longer runtime.
    p_value: the probability of falsely detecting correlated signal at a given slowness in random 
         noise; used to determine the threshold F statistic for stopping. Note that because many
         slownesses are tested, not just one, the actual probability of false positives is higher.
    
    nWelch: number of non-overlapping windows used to calculate cross-spectrum.
    freq_bin_width: number of adjacent samples in spectrum contained in each frequency bin
    freq_min: lowest frequency used in calculations (Hz). This is not a filter's corner frequency.

    freq_max: highest frequency used in calculations (Hz). This is not a filter's corner frequency.

    separate_freqs: method used to determine peak of frequency-slowness spectrum in each iteration.
         Only '0' is currently recommended.
         --0 (default): sum power over all frequencies
         --1: treat each frequency-slowness combination separately
         --2: try to intelligently consider only frequencies that seem coherent at the same slowness
    rawFT: if True, do not apply tapers or detrending when calculating spectrum and cross-spectrum
    taper: used to taper time windows before calculating spectrum. Can be either default 'Tukey' or 
         'multitaper', or a numpy.array containing the user-defined taper.

    taper_param: unused for Tukey; refers to the time-bandwidth product for multitaper.
    prewhiten: if True, prewhiten spectra before calculating cross-spectrum
    show_plots: if True, show a slowness spectrum plot each iteration
    verbose: if True, give verbose output
    steering_vectors: if not None, has dimensions [f,nx,ny,nsta]
    cross_spec: if the cross spectrum has already been calculated, it can be provided to save time.
    cross_spec_mask: matrix of zeros and ones indicating which station pairs should be considered. 
    By default, all station pairs are used; however, the user might want to only consider station 
    pairs within some maximum distance.

    Returns
    -------
    Dict including clean frequency-slowness spectrum, original frequency-slowness spectrum and 
    cross-spectrum, frequency-slowness spectrum and cross-spectrum of what's left after cleaning,
    and frequency/slowness axes for the outputs. cross-spectra indices are [chan_i, chan_j, freq],
    and freq-slowness spectra have indices [freq, x_slow, y_slow].

    See the demo files for examples demonstrating use.
    """
    staLoc, sxList, syList = _process_inputs(stream, x, y, sxList, syList)
    nsta = staLoc.shape[0]    

    if (cross_spec is None) or (freqList is None):
        ## calculate the cross-spectrum
        if verbose: print('Calculating cross-spectrum')
        cross_spec, FT, freqList, dfN, dfD = calc_cross_spectrum(stream, raw = rawFT, win_length_sec = win_length_sec,
                                                                freq_bin_width = freq_bin_width, prewhiten = prewhiten,
                                                                taper = taper, taper_param = taper_param, criterion_function = criterion_function)
    ## drop freqs outside the user-defined range
    w = np.where((freqList <= freq_max) & (freqList >= freq_min))[0]
    cross_spec = cross_spec[:,:,w]
    freqList = freqList[w]

    if cross_spec_mask is None:
        cross_spec_mask = np.ones((nsta, nsta))
    
    ####### In calc_fourier_window or calc_cross_spectrum, need to consider the number of freqs as Z when defining dfN and dfD
    ## calculate the stopping F value
    if separate_freqs == 0:
        dfN *= len(freqList) 
        dfD *= len(freqList)
    stopF = scipy.stats.f.ppf(1-p_value, dfN, dfD)

    ## calculate the steering vectors and weights
    if verbose: print('Calculating steering vectors and weights')
    if steering_vectors is None:
        steering_vectors = make_steering_vectors(stream, freqList, sxList, syList)

    wB = calc_weights(steering_vectors)
    ## save the spectra before cleaning
    originalCrossSpec = cross_spec.copy()
    originalSpec = np.abs(np.einsum('klmi,ijk,klmj->klm', wB.conj(), cross_spec, wB)) # n_freq x n_slow x n_slow; absolute power over f, sx,sy
    ## Run the cleaning loop
    if verbose:
        print('Beginning cleaning loop')
        print('[sxList[imax], syList[jmax], Pmax/stopCondition]')
    cleanSpec = np.zeros(wB.shape[:3])
    clean_output = clean_step(cleanSpec, cross_spec, wB, phi, stopF, separate_freqs, verbose, sxList, syList, show_plots, cross_spec_mask)
    cleanSpec, cross_spec, remainingSpec, droppedCrossSpec = [clean_output[i] for i in ['cleanSpec', 'cross_spec', 'remainingSpec', 'droppedCrossSpec']]

    ## Print the remaining power and original power, and return
    print('Remaining power fraction: %.3f' % np.abs(np.einsum('iij->', cross_spec) / np.einsum('iij->', originalCrossSpec)))
    return {'cleanSpec':cleanSpec, 'originalSpec':originalSpec, 'remainingSpec':remainingSpec, 'remainingCrossSpec':cross_spec, 
            'originalCrossSpec':originalCrossSpec, 'droppedCrossSpec':droppedCrossSpec, 'sx':sxList, 'sy':syList, 'freq':freqList}

#####################
def _process_inputs(stream, x = None, y = None, sxList = None, syList = None):
    """
    Validate and apply defaults to clean inputs.
    """
    if len(stream) == 0:
        raise Exception('stream has no traces')
    if any([len(stream[0].data) != len(tr.data) for tr in stream]):
        raise Exception('traces are not all the same length')
    if len(stream[0].data) == 0:
        raise Exception('traces in stream are empty')
    
    ## work out the station geometry
    if (x is None) or (y is None):
        try:
            staLoc = obspy.signal.array_analysis.get_geometry(stream) # https://docs.obspy.org/packages/autogen/obspy.signal.array_analysis.get_geometry.html
        except:
            staLoc = obspy.signal.array_analysis.get_geometry(stream, 'xy')
    else:
        staLoc = np.array([x,y])
    
    ## work out the slownesses to test
    defaultSlownessList = np.arange(-0.004, 0.004, 0.0001) * 1000 # sec/km, for obspy consistency
    if sxList is None: 
        sxList = defaultSlownessList
    else: 
        sxList = np.array(sxList)
    if syList is None:  
        syList = defaultSlownessList
    else: 
        syList = np.array(syList)

    return (staLoc, sxList, syList)
###########################################


   
def calc_semblance(clean_output, type = 'original'):
    total_power = np.real(np.einsum('iik->', clean_output[type.lower() + 'CrossSpec']))
    return np.einsum('ijk->jk', clean_output[type.lower() + 'Spec'] / total_power)

def check_output_power(clean_output):
    """
    Verify that power is conserved by the cleaning operation. Original power from the cross-spectrum
    diagonal should be equal to the sum of the output clean power and remaining un-clean power.
    
    Parameters
    ----------
    clean_output: dict, output of clean()

    Result: ratio of total output to input power; should be very close to 1
    """
    original_power = np.einsum('iij ->', clean_output['originalCrossSpec'])
    remaining_power = np.einsum('iij ->', clean_output['remainingCrossSpec'])
    clean_power = np.sum(clean_output['cleanSpec'])
    
    ratio = np.real((clean_power + remaining_power)/original_power)
    print(ratio)
    return np.abs(ratio - 1) < 1e-3


 
def add_inv_coords(st, inv):
    """
    For each trace tr in stream st, adds a 'coordinates' AttribDict to tr.stats using location info
    from an inventory. See application in obspy.signal.array_analysis.array_processing.

    Parameters
    ----------
    st: stream that locations should be added to
    inv: obspy.Inventory or pandas.DataFrame containing locations for all traces in st

    Returns: None, changes st in place
    """
    if type(inv) is obspy.Inventory: 
        for tr in st:
            loc = obspy.core.AttribDict(inv.get_coordinates(tr.get_id()))
            tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    elif type(inv) is pd.DataFrame:
        for tr in st:
            id = tr.get_id()
            w = np.where(inv.SN == tr.stats.station)[0][0]
            loc = obspy.core.AttribDict(
                {'latitude':inv.loc[w,'lat'],
                 'longitude':inv.loc[w,'lon'],
                 'elevation':0,
                 'local_depth':0}
                )
            tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    
############

######################
def backproject(lon_grid_range, lat_grid_range, grid_spacing_deg, lon_station, lat_station, lon_eq, lat_eq, z_eq, ca_min, ca_max, cs_min, cs_max, baz_arrival, time_arrival, power_arrival = None, az_bin_width = 10):
    eps = 1e-6
    if power_arrival is None:
        power_arrival = np.ones(len(time_arrival))
    
    ## Make map matrix. Map consists of cells, with nodes at the cell corners.
    #lon_nodes = np.arange(lon_grid_range[0], lon_grid_range[1], grid_spacing_deg)
    #lat_nodes = np.arange(lat_grid_range[0], lat_grid_range[1], grid_spacing_deg)

    ## For each node, calculate max time and min time, and max baz and min baz.
    #x_hat_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    #y_hat_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    #t_max_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    #t_min_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    #for i, lon in enumerate(lon_nodes):
    #    for j, lat in enumerate(lat_nodes):
    #        [ac_dist, ac_az, ac_baz] = obspy.geodetics.gps2dist_azimuth(lat1=lat, lon1=lon, lat2=lat_station, lon2=lon_station)
    #        [seis_dist, seis_az, seis_baz] = obspy.geodetics.gps2dist_azimuth(lat1=lat, lon1=lon, lat2=lat_eq, lon2=lon_eq)
    #        x_hat_nodes[i,j] = np.sin(ac_baz * np.pi/180)
    #        y_hat_nodes[i,j] = np.cos(ac_baz * np.pi/180)
    #        t_max_nodes[i,j] = ac_dist/ca_min + np.sqrt(seis_dist**2 + z_eq**2)/cs_min
    #        t_min_nodes[i,j] = ac_dist/ca_max + np.sqrt(seis_dist**2 + z_eq**2)/cs_max
    (lon_nodes, lat_nodes, x_hat_nodes, y_hat_nodes, t_max_nodes, t_min_nodes) = calc_map_node_traveltime(lon_grid_range, lat_grid_range, grid_spacing_deg, lon_station, lat_station, lon_eq, lat_eq, z_eq, ca_min, ca_max, cs_min, cs_max)
    
    ## make cells, with max/min values for boundary nodes
    x_hat_min_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    x_hat_max_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    y_hat_min_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    y_hat_max_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    t_max_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    t_min_cells = np.zeros((len(lon_nodes)-1, len(lat_nodes)-1))
    for i in np.arange(len(lon_nodes) - 1):
        for j in np.arange(len(lat_nodes) - 1):
            x_hat_min_cells[i,j] = np.min(x_hat_nodes[i:(i+2), j:(j+2)])
            x_hat_max_cells[i,j] = np.max(x_hat_nodes[i:(i+2), j:(j+2)])
            y_hat_min_cells[i,j] = np.min(y_hat_nodes[i:(i+2), j:(j+2)])
            y_hat_max_cells[i,j] = np.max(y_hat_nodes[i:(i+2), j:(j+2)])
            t_min_cells[i,j] = np.min(t_min_nodes[i:(i+2), j:(j+2)])
            t_max_cells[i,j] = np.min(t_max_nodes[i:(i+2), j:(j+2)])
    ## For each detection, assign its power to output cells
    output_cells = 0 * t_max_cells
    bin = np.linspace(-0.5 * az_bin_width, 0.5 * az_bin_width + eps, 10)
    for time, baz, power in zip(time_arrival, baz_arrival, power_arrival):
        #### identify and mark cells that contain baz and t
        x_hat_min = np.min(np.sin((baz + bin) * np.pi/180))
        x_hat_max = np.max(np.sin((baz + bin) * np.pi/180))
        y_hat_min = np.min(np.cos((baz + bin) * np.pi/180))
        y_hat_max = np.max(np.cos((baz + bin) * np.pi/180))
        output_cells += power * ( (time >= t_min_cells) & (time <= t_max_cells) &
                                  (x_hat_max >= (x_hat_min_cells-eps)) & (x_hat_min <= (x_hat_max_cells+eps)) &
                                  (y_hat_max >= (y_hat_min_cells-eps)) & (y_hat_min <= (y_hat_max_cells+eps)) )

    return output_cells, lon_nodes, lat_nodes, t_min_nodes, t_max_nodes


#############
def calc_map_node_traveltime(lon_grid_range, lat_grid_range, grid_spacing_deg, lon_station, lat_station, lon_eq, lat_eq, z_eq, ca_min, ca_max, cs_min, cs_max):
    ## Make map matrix. Map consists of cells, with nodes at the cell corners.
    lon_nodes = np.arange(lon_grid_range[0], lon_grid_range[1], grid_spacing_deg)
    lat_nodes = np.arange(lat_grid_range[0], lat_grid_range[1], grid_spacing_deg)

    ## For each node, calculate max time and min time, and max baz and min baz.
    x_hat_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    y_hat_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    t_max_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    t_min_nodes = np.zeros((len(lon_nodes), len(lat_nodes)))
    for i, lon in enumerate(lon_nodes):
        for j, lat in enumerate(lat_nodes):
            [ac_dist, ac_az, ac_baz] = obspy.geodetics.gps2dist_azimuth(lat1=lat, lon1=lon, lat2=lat_station, lon2=lon_station)
            [seis_dist, seis_az, seis_baz] = obspy.geodetics.gps2dist_azimuth(lat1=lat, lon1=lon, lat2=lat_eq, lon2=lon_eq)
            x_hat_nodes[i,j] = np.sin(ac_baz * np.pi/180)
            y_hat_nodes[i,j] = np.cos(ac_baz * np.pi/180)
            t_max_nodes[i,j] = ac_dist/ca_min + np.sqrt(seis_dist**2 + z_eq**2)/cs_min
            t_min_nodes[i,j] = ac_dist/ca_max + np.sqrt(seis_dist**2 + z_eq**2)/cs_max
    return (lon_nodes, lat_nodes, x_hat_nodes, y_hat_nodes, t_max_nodes, t_min_nodes)




#############
## no longer needed?
def add_xy_coords(st, x, y, z = None):
    """
    For each trace tr in stream st, adds a 'coordinates' AttribDict to tr.stats using location info
    from user-provided cartesian coordinate arrays. See application in 
    obspy.signal.array_analysis.array_processing.

    Parameters
    ----------
    st: stream that locations should be added to
    inv: inventory containing locations for all traces in st

    Returns: None, changes st in place
    """
    x = np.array(x)
    y = np.array(y)
    if z is None:
        z = x * 0
    for i, tr in enumerate(st):
        loc = obspy.core.AttribDict()
        loc['x'] = x[i]
        loc['y'] = y[i]
        loc['elevation'] = z[i]
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    return st

def add_latlon_coords(st, lat, lon, z = None):
    """
    For each trace tr in stream st, adds a 'coordinates' AttribDict to tr.stats using location info
    from user-provided lat/lon coordinate arrays. See application in 
    obspy.signal.array_analysis.array_processing.

    Parameters
    ----------
    st: stream that locations should be added to
    inv: inventory containing locations for all traces in st

    Returns: None, changes st in place
    """
    lat = np.array(lat)
    lon = np.array(lon)
    if z is None:
        z = lat * 0
    for i, tr in enumerate(st):
        loc = obspy.core.AttribDict()
        loc['latitude'] = lat[i]
        loc['longitude'] = lon[i]
        loc['elevation'] = z[i]
        tr.stats = obspy.core.Stats({**tr.stats, 'coordinates': loc})
    return st

