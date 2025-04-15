import PSCalib.GlobalUtils as gu

def str_to_int_or_float(s):
    """
    Checks if s is a float or int and converts to corresponding type.

    Parameters
    ----------
    s : str
        Input string to check
    """
    v = float(s)
    if v%1 == 0: v=int(v)
    return v

def sfields_to_xyz_vector(flds):
    """ 
    Converts ['+0.002583x', '-0.999997y', '+0.000000z'] to (0.002583, -0.999997, 0.000000)

    Parameters
    ----------
    flds : list
        List of strings representing x, y, and z coordinates
    """
    v = (float(flds[0].strip('x')), float(flds[1].strip('y')))
    z = float(flds[2].strip('z')) if len(flds)==3 else 0
    v += (z,)
    return v

def str_is_segment_and_asic(s):
    """ 
    Checks if s looks like str 'q0a2' or 'p12a7' and
    returns 'p0.2' or 'p12.7' or False

    Parameters
    ----------
    s : str
        Input string to check
    """
    if not isinstance(s, str)\
    or len(s)<2: return False
    flds = s[1:].split('a')
    return False if len(flds) !=2 else\
        'p%sa%s' % (flds[0], flds[1]) if all([f.isdigit() for f in flds]) else\
        False

def header_psana(det_type):
    """
    Returns the header for the psana geometry file based on the detector type.
    The header includes information such as the title, date, author, experiment,
    detector, calibration type, and comments.

    Parameters
    ----------
    det_type : str
        The type of detector (e.g., 'Rayonix', 'ePix10k2M', etc.)
    """
    if det_type.lower() == 'rayonix':
        txt=\
        '# TITLE       Geometry parameters of Rayonix'\
        +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
        +'\n# METROLOGY  no metrology available'\
        +'\n# AUTHOR     %s' % gu.get_login()\
        +'\n# EXPERIMENT N\A'\
        +'\n# DETECTOR   Rayonix'\
        +'\n# CALIB_TYPE geometry'\
        +'\n# COMMENT:01 Automatically created from BayFAI for the Rayonix detector'\
        +'\n'
    elif det_type.lower() == 'epix10k2m':
        txt=\
        '# TITLE       Geometry parameters of ePix10k2M'\
        +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
        +'\n# METROLOGY  no metrology available'\
        +'\n# AUTHOR     %s' % gu.get_login()\
        +'\n# EXPERIMENT N\A'\
        +'\n# DETECTOR   Epix10ka2M'\
        +'\n# CALIB_TYPE geometry'\
        +'\n# COMMENT:01 Automatically created from BayFAI for the 16-segment ePix10k2M detector'\
        +'\n'
    elif 'epix10kaquad' in det_type.lower():
        txt=\
        '# TITLE       Geometry parameters of ePix10kaQuad'\
        +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
        +'\n# METROLOGY  no metrology available'\
        +'\n# AUTHOR     %s' % gu.get_login()\
        +'\n# EXPERIMENT N\A'\
        +'\n# DETECTOR   %s' % det_type\
        +'\n# CALIB_TYPE geometry'\
        +'\n# COMMENT:01 Automatically created from BayFAI for the 8-segment ePix10kaQuad detector'\
        +'\n'
    elif det_type.lower() == 'jungfrau4m':
        txt=\
        '# TITLE       Geometry parameters of Jungfrau'\
        +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
        +'\n# METROLOGY  no metrology available'\
        +'\n# AUTHOR     %s' % gu.get_login()\
        +'\n# EXPERIMENT N\A'\
        +'\n# DETECTOR   DetLab.0:Jungfrau.2 or jungfrau4M'\
        +'\n# CALIB_TYPE geometry'\
        +'\n# COMMENT:01 Automatically created from BayFAI for the 8-segment Jungfrau4M detector'\
        +'\n'
    txt +=\
        '\n# PARAM:01 PARENT     - name and version of the parent object'\
        '\n# PARAM:02 PARENT_IND - index of the parent object'\
        '\n# PARAM:03 OBJECT     - name and version of the object'\
        '\n# PARAM:04 OBJECT_IND - index of the new object'\
        '\n# PARAM:05 X0         - x-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:06 Y0         - y-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:07 Z0         - z-coordinate [um] of the object origin in the parent frame'\
        '\n# PARAM:08 ROT_Z      - object design rotation angle [deg] around Z axis of the parent frame'\
        '\n# PARAM:09 ROT_Y      - object design rotation angle [deg] around Y axis of the parent frame'\
        '\n# PARAM:10 ROT_X      - object design rotation angle [deg] around X axis of the parent frame'\
        '\n# PARAM:11 TILT_Z     - object tilt angle [deg] around Z axis of the parent frame'\
        '\n# PARAM:12 TILT_Y     - object tilt angle [deg] around Y axis of the parent frame'\
        '\n# PARAM:13 TILT_X     - object tilt angle [deg] around X axis of the parent frame'\
        '\n'\
        '\n# HDR PARENT IND     OBJECT IND    X0[um]   Y0[um]   Z0[um]   ROT-Z  ROT-Y  ROT-X     TILT-Z    TILT-Y    TILT-X'
    return txt