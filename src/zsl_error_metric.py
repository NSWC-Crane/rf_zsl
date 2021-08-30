import numpy as np

def zsl_error_metric(x, y):

    """
    function to calculate the magnitude and phase metrices between an
    original dataset and a reconstructed dataset

    Parameters
    ----------
    x : numpy array
        array containing the original values packed in IQIQIQ format
    y : numpy array
        array containing the reconstructed values packed in IQIQIQ format

    Returns
    -------
    dist_mean : float64
        mean of the complex distance between points within x and y
    dist_std : float64
        standard deviation of the complex distance between points within x and y
    phase_mean : float64
        mean of the phase angle between points within x and y
    phase_std : float64
        standard deviation of the complex distance between points within x and y
    """
    # convert x into a complex numpy array
    x = x.reshape(-1, 2)

    xc = np.empty(x.shape[0], dtype=complex)
    xc.real = x[:, 0]
    xc.imag = x[:, 1]

    # convert y into a complex numpy array
    y = y.reshape(-1, 2)

    yc = np.empty(y.shape[0], dtype=complex)
    yc.real = y[:, 0]
    yc.imag = y[:, 1]

    # calculate the distance error
    dist = np.absolute(xc-yc)
    dist_mean = np.mean(dist)
    dist_std = np.std(dist)

    # compute that phases for each
    ang_xc = np.angle(xc, deg=True)
    ang_yc = np.angle(yc, deg=True)

    # remove any zero angles in favor of 360 degrees
    # ang_xc[ang_xc == 0] = 360
    # ang_yc[ang_yc == 0] = 360

    # compute the ratio of the phase difference
    phase_diff = (ang_xc - ang_yc)

    # compute the mean and std of the phase error
    phase_mean = np.mean(phase_diff)
    phase_std = np.std(phase_diff)

    bp = 1

    return dist_mean, dist_std, phase_mean, phase_std
