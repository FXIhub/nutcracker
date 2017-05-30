import numpy as np
import scipy as sp

def envelope(function,frequence,peak_finding_threshold=(None,None),order_spline_interpolation=3):
    """
    Calculates the upper and lower envelope of a given function.
    This function is based on https://stackoverflow.com/questions/34235530/python-how-to-get-high-and-low-envelope-of-a-signal (24.05.2017)

    Args:
        :function(float ndarray):         1d ndarray containing the function values
        :frequence(int):                  frequence at which function should be sample

    Kwargs:
        :peak_finding_threshold(list):    threshold of peak finder given as tuple (min,max), peaks would then not take into account, default=(None,None)
        :order_spline_interpolation(int): order of the spline interpolation
    """

    # creating lists for envelope
    upper_envelope_x = []
    upper_envelope_y = []

    lower_envelope_x = []
    lower_envelope_y = []

    # iterating through function at given frequency
    for k in np.arange(0,len(function), frequence):
        part = function[k:k+frequence]

        upper_envelope_x.append(k)
        upper_envelope_y.append(part.max())

        lower_envelope_x.append(k)
        lower_envelope_y.append(part.min())

    # transform to numpy arrays
    upper_envelope_x = np.array(upper_envelope_x)
    upper_envelope_y = np.array(upper_envelope_y)
    lower_envelope_x = np.array(lower_envelope_x)
    lower_envelope_y = np.array(lower_envelope_y)

    # peak finding lower envelope
    if peak_finding_threshold[0]:
        for k in np.arange(len(lower_envelope_y)):
            if lower_envelope_y[k] < peak_finding_threshold[0]:
                lower_envelope_y[k] = np.mean([lower_envelope_y[k-2],lower_envelope_y[k-1]])

    # peak finding upper envelope
    if peak_finding_threshold[1]:
        for k in np.arange(len(upper_envelope_y)):
            if upper_envelope_y[k] > peak_finding_threshold[1]:
                upper_envelope_y[k] = np.mean([upper_envelope_y[k-2],upper_envelope_y[k-1]])

    # interpolate the envelope function
    upper_envelope = sp.interpolate.interp1d(upper_envelope_x,upper_envelope_y, kind = order_spline_interpolation,bounds_error = False, fill_value=0.0)
    lower_envelope = sp.interpolate.interp1d(lower_envelope_x,lower_envelope_y, kind = order_spline_interpolation,bounds_error = False, fill_value=0.0)

    x = np.arange(len(function))

    upper_envelope = upper_envelope(x)
    lower_envelope = lower_envelope(x)

    return upper_envelope, lower_envelope
