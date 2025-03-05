import numpy as np

def get_beam_center(params):
    """
    From pyFAI calibration parameters, return the beam center in meters

    Parameters
    ----------
    params : list
        Detector parameters found by PyFAI calibration
    """
    dist = params[0]
    poni1 = params[1]
    poni2 = params[2]
    rot1 = params[3]
    rot2 = params[4]
    distance = dist / (np.cos(rot1) * np.cos(rot2))
    cx = poni1 + dist * np.tan(rot2) / np.cos(rot1)
    cy = poni2 - dist * np.tan(rot1)
    return distance, cx, cy