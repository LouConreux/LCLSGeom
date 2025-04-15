import numpy as np
from math import degrees, atan2

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

def angle_and_tilt(a):
    """
    From a given angle, return the angle and tilt in degrees

    Parameters
    ----------
    a : float
        Angle in degrees
    """
    desangles = np.array((-180,-90, 0, 90, 180))
    difangles = a-desangles
    absdifang = np.abs(difangles)
    imin = np.where(absdifang == np.amin(absdifang))[0]
    angle, tilt = desangles[imin], difangles[imin]
    return (angle if angle>=0 else angle+360), tilt

def unit_vector_pitch_angle_max_ind(u):
    """
    From a given unit vector, return the pitch angle and index of maximum value
    in the vector

    Parameters
    ----------
    u : numpy.ndarray
        Input unit vector
    """
    absu = np.absolute(u)
    imax = np.where(absu == np.amax(absu))[0]
    pitch = degrees(atan2(u[2],u[imax]))
    pitch = (pitch+180) if pitch<-90 else (pitch-180) if pitch>90 else pitch
    return pitch, imax

def tilt_xy(uf, us):
    """
    From two unit vectors, return the tilt in x and y directions
    based on the maximum index

    Parameters
    ----------
    uf : numpy.ndarray
        First unit vector
    us : numpy.ndarray
        Second unit vector
    """
    tilt_f, imaxf = unit_vector_pitch_angle_max_ind(uf)
    tilt_s, imaxs = unit_vector_pitch_angle_max_ind(us)
    tilt_x, tilt_y = (tilt_s, tilt_f) if imaxf==0 else (tilt_f, tilt_s)
    return tilt_x, -tilt_y

def rotate_z(angle_z, angle_x, angle_y):
    """
    For a given angle around Z-axis, switch angle_x and angle_y appropriately.

    Parameters
    ----------
    angle_z : float
        Angle around Z-axis in degrees in [0, 90, 180, 270]
    angle_x : float
        Angle around X-axis in degrees in [0, 90, 180, 270]
    angle_y : float
        Angle around Y-axis in degrees in [0, 90, 180, 270]
    """
    if angle_z == 0:
        return angle_x, angle_y
    elif angle_z == 90:
        return angle_y, -angle_x
    elif angle_z == 180:
        return -angle_x, -angle_y
    elif angle_z == 270:
        return -angle_y, angle_x