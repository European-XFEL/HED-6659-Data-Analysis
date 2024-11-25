import cmath

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import chi2


def find_shocks(
    ref: np.ndarray,
    shot: np.ndarray,
    time: np.ndarray,
    roi_ref: tuple,
    roi_phase: tuple,
    mult: float = -1,
    W: int = 20,
    F: int = 2,
    p_thresh: float = 5e-4,
    ds: int = 100
) -> np.ndarray:
    """
    Find shock breakout in VISAR data.

    Args:
        ref: Reference image.
        shot: Shot image.
        time: Time array corresponding to the image width.
        ref_roi: Region of interest for reference (left, width, top, height).
        phase_roi: Region of interest for phase (left, width, top, height).
        mult: Multiplier applied to retrieved phase (default: -1).
        W: Width of backward-looking window in pixels (default: 20).
        F: Width of forward-looking window in pixels (default: 2).
        p_thresh: Threshold for shock detection algorithm p-value (default: 5e-4).
        ds: Number of downsampled points for shock search (default: 100).

    Returns:
        Array of shock breakout times.
    """

    #Find fringe carrier wavelength (used to define FFT filter)
    refroi_im = ref[roi_ref]
    refroi_im_mean = np.mean(refroi_im, axis=1)
    carrier_wave   = find_carrier(refroi_im_mean)

    #Analyse phase roi
    phaseroi_shot = shot[roi_phase]

    #Use FFT method to find space averaged phase and amplitude
    phase_shot, amplitude_shot = analyze_roi(phaseroi_shot, carrier_wave)

    #Subtract reference fringe phase
    phaseroi_ref = ref[roi_phase]
    phase_ref, _ = analyze_roi(phaseroi_ref, carrier_wave)
    phase = (phase_shot - phase_ref) * mult

    #Downsample phase and amplitude signals priot to shock search
    time_roi = time[roi_phase[1]]
    tmin     = time_roi[0]
    tmax     = time_roi[-1]

    time_ds  = np.linspace(tmin, tmax, num=ds)

    phase_ds          = np.interp(time_ds, time_roi, phase)
    amplitude_shot_ds = np.interp(time_ds, time_roi, amplitude_shot)

    #Run statistical algorithm on phase
    t_list, S_t_list, p_t_list = p_t(phase_ds, W, F, 'UP')

    #Find shock locations from p_t crossings
    phase_shocklist = find_crossings(t_list, p_t_list, p_thresh)

    #Run statistical algorithm on amplitude
    t_list, S_t_list, p_t_list = p_t(amplitude_shot_ds, W, F, 'UP_OR_DOWN')

    #Find shock locations from p_t crossings
    amplitude_shocklist = find_crossings(t_list, p_t_list, p_thresh)

    shocklist = np.append(phase_shocklist, amplitude_shocklist)

    if len(shocklist) == 0:
        print('No shocks found')
        return np.array([])

    #Clean up list
    tolerance    = 2*F
    shocklist, _ = clean_list(shocklist, tolerance)

    #Return shocktime(s)
    shocktimes = time_ds[shocklist]

    return shocktimes

#***********************************************************************************************
def find_carrier(lineout):

    n_points = len(lineout)

    #---------------------------------------------------------------
    #Find approximate carrier wavelength from FFT spectrum
    #---------------------------------------------------------------
    lineout_fft     = np.fft.fft(lineout)
    freq            = np.fft.fftfreq(lineout.shape[-1])
    flipped         = np.abs(lineout_fft[:n_points//2:-1])
    carrier_loc     = np.argmax(flipped) + 1
    carrier_wave_fft = round(1./freq[carrier_loc])

    return carrier_wave_fft

#***********************************************************************************************
def analyze_roi(phase, carrier_lamda):

    phase = median_filter(phase, size=5)

    # Clip height to have integer number of fringes
    nfringes = int(phase.shape[0] / carrier_lamda)

    # Define truncated ROI
    n_rows = round(nfringes * carrier_lamda)
    trunc = phase[0:n_rows, :]

    n_cols = phase.shape[1]

    if n_rows == 0: #Failed to find carrier wavelength
        print('Failed to find fringe carrier wavelength. Check fringe contrast and/or reference ROI')
        return

    # Find carrier location in FFT array
    carrier_loc = round(n_rows / carrier_lamda)

    # Define FFT filter
    f_min = carrier_loc - 1
    f_max = carrier_loc + 1
    fft_filter = [(f >= f_min and f <= f_max) for f in np.arange(n_rows)]

    # Retrieve amplitude and phase
    roi_filtered = []
    for index in np.arange(n_cols):
        col          = trunc[:, index]        
        col_fft      = np.fft.fft(col)    
        col_fft_filt = col_fft * fft_filter
        col_ifft     = np.fft.ifft(col_fft_filt)

        roi_filtered.append(col_ifft)

    # Transpose image to right way round
    roi_filtered = np.transpose(roi_filtered) #Complex numbers

    # Record amplitude
    amplitude = np.mean(np.abs(roi_filtered), axis=0)

    phase = []

    for row in roi_filtered:
        shifted_row = np.roll(row, -1)
        phi_row     = np.zeros(n_cols)

        for n in np.arange(n_cols):
            ratio       = shifted_row[n] / row[n]
            dphi        = cmath.phase(ratio)
            phi_row[n]  = dphi

        phi_row    = np.roll(phi_row, 1)
        phi_row[0] = 0.0

        phi_cumsum = np.cumsum(phi_row)

        phase.append(phi_cumsum)

    #Unwrap phase image in time direction
    phase = np.unwrap(phase, axis=1)

    #Unwrap phase image in the spatial direction
    #phase = np.unwrap(phase, axis=0)

    #Average in spatial direction    
    phase = np.mean(phase, axis=0)

    return phase, amplitude


# Jonty's statistical algorithm
def p_t(sig_t, W, F, change_dir):

    siglength = len(sig_t)
    delta_t = np.diff(sig_t) #first element is delta_1...

    #Stats tracked
    S_t_list = []
    t_list   = np.arange(W, siglength-F)
    p_t_list = []

    for t in np.arange(W, siglength-F):
        # S_t
        # Null hypothesis
        past    = delta_t[t-W:t]
        future  = delta_t[t:t+F]

        delta_0 = np.append(past, future)        
        muhat_0 = np.mean(delta_0)
        RSS_0   = np.sum(np.square(delta_0 - muhat_0))
        s2hat_0 = RSS_0 / (W+F)

        # G stat
        muhat_W = np.mean(past)
        RSS_W   = np.sum(np.square(past - muhat_W))

        muhat_F = np.mean(future)
        RSS_F   = np.sum(np.square(future - muhat_F))

        s2hat   = (RSS_W + RSS_F) / (W+F)

        # Adapt if only looking for positive changes    
        if muhat_F <= muhat_W and change_dir == 'UP':
            s2hat = s2hat_0

        S_t = (W+F) * (np.log(s2hat_0)-np.log(s2hat))

        # Record S_t
        S_t_list.append(S_t)

        # p_t
        p_t = 1. - chi2.cdf(x=S_t, df=1)

        #Record p_t
        p_t_list.append(p_t)

    return t_list, S_t_list, p_t_list


#***********************************************************************************************
#Find negative going zero crossings in p_t
#***********************************************************************************************
def find_crossings(t_list, p_t, p_thresh):

    threshcrossed = np.where(np.array(p_t) < p_thresh, 1, 0)
    crossings     = np.where(np.diff(threshcrossed, prepend=0)==1)
    t_shock_list  = t_list[crossings[0]]

    return t_shock_list

#***********************************************************************************************
#Clean up list of shock locations (discovered by shock detection algorithm)
#***********************************************************************************************
def clean_list(shocklist, tolerance):

    shocklist_sorted = np.sort(shocklist)

    diff = (np.abs(np.diff(shocklist_sorted)) > tolerance).astype(int)

    index = np.where(diff==1)[0]

    start = 0

    cleaned_list = []
    weights_list = []

    for group in index:

        end = group + 1

        current_group = shocklist_sorted[start:end]

        cleaned_list.append(round(np.mean(current_group)))

        #Weight: how many time shocks found
        current_weight = len(current_group)
        weights_list.append(current_weight)

        start = end

    #Last group
    last_group = shocklist_sorted[start:]

    cleaned_list.append(round(np.mean(last_group)))

    last_weight = len(last_group)
    weights_list.append(last_weight)

    return cleaned_list, weights_list

#***********************************************************************************************
#Used for visualizing ROI only. Can be omitted
#***********************************************************************************************
def show_roi(image, roi, name):
    import matplotlib.patches as patches

    y_start, y_end = roi[0].start, roi[0].stop
    x_start, x_end = roi[1].start, roi[1].stop
    width = x_end - x_start
    height = y_end - y_start

    fig, ax = plt.subplots(figsize = (6,6))
    ax.imshow(image)
    # Add the rectangle
    rectangle = patches.Rectangle(
        (x_start, y_start), width, height, edgecolor='blue', facecolor='none', linewidth=2
    )
    ax.add_patch(rectangle)

    ax.set_title(name)
    ax.set_xlabel('Time (pix)')
    ax.set_ylabel('Position (pix)')

    # plt.show()

    # plt.savefig(name + '_roi')
