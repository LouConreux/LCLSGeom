"""
LCLS-II detectors:
File manager functions to handle push and pull of geometry files to and from the psana calibration database.
This module also includes functions to fetch default geometry files if not found in the calibration database.

LCLS-I detectors:
File manager functions to look for geometry in the experiment calibration directory.
This module also includes functions to fetch default geometry files if not found in the experiment calibration directory.
"""

import logging

import psana
if hasattr(psana, "xtc_version"):
    from psana import DataSource
    import psana.pscalib.calib.MDBUtils as mu
    import psana.pscalib.calib.MDBWebUtils as wu
    import psana.detector.UtilsCalib as uc
    cc = wu.cc

    IS_PSANA2 = True
else:
    IS_PSANA2 = False

import os
import re

from LCLSGeom.calib import group_from_detname, source_from_detname, select_calib_file, clean_detname

from logging import getLogger

logger: logging.Logger = getLogger(__name__)


def get_geometry(exp: str, run: int, detname: str, skip_load: bool = False) -> str:
    """
    Get the geometry file for a given experiment, run, and detector name. 
    If LCLS-II, check first the calibration database, then the default geometry file.
    If LCLS-I, check first the experiment calibration directory, then the default geometry file.
    If not found in either location, get default geometry file from templates.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detname : str
        Name of the detector
    skip_load : bool, optional
        If True, skip loading the geometry file and return the default geometry file path. Default is False.
    
    Returns
    -------
    in_file : str
        Path to the geometry file
    """
    try:
        if IS_PSANA2:
            in_file = pull(exp=exp, run=run, detname=detname, skip_load=skip_load)
            logger.info(f"Geometry file for {detname} found in calibration database for experiment {exp} and run {run}.")
        else:
            in_file = check(exp=exp, run=run, detname=detname)
            logger.info(f"Geometry file for {detname} found in experiment calibration directory for experiment {exp} and run {run}.")
        return in_file
    except FileNotFoundError as e:
        logger.warning(str(e))
        logger.info(f"Fetching default geometry file for {detname}.")
        in_file = get_default_geometry(detname)
        return in_file


def get_default_geometry(detname: str) -> str:
    """
    Get the default geometry file for a given detector name.
    For LCLS-II, check first the psana installation for the default geometry file, then fetch from templates if not found.

    Parameters
    ----------
    detname : str
        Name of the detector
    
    Returns
    -------
    in_file : str
        Path to the default geometry file
    """
    detname = clean_detname(detname)
    if IS_PSANA2:
        psana_path = psana.__path__[0]
        data_dir = os.path.join(psana_path, "pscalib", "geometry", "data")
        in_file = os.path.join(data_dir, f"geometry-def-{detname}.data")
        if not os.path.exists(in_file):
            logger.warning(f"Default geometry file not found for {detname} in psana installation. Fetching from templates.")
            in_file = fetch_template(detname)

    else:
        in_file = fetch_template(detname)

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Default geometry file not found for detector {detname}.")

    return in_file


def pull(exp: str, run: int, detname: str, skip_load: bool = False) -> str:
    """
    If LCLS-II, pull the geometry file for a given experiment, run, and detector name from the calibration database.
    If LCLS-I, pull the geometry file for a given experiment, run, and detector name from the experiment calibration directory.
    
    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detname : str
        Name of the detector
    skip_load : bool, optional
        If True, skip loading the geometry file and return the default geometry file path. Default is False.
    """
    if skip_load:
        ds = DataSource(exp=exp, run=run, skip_calib_load="all")
    else:
        ds = DataSource(exp=exp, run=run)
    runs = next(ds.runs())
    detector = runs.Detector(detname)
    longname: str = detector.raw._uniqueid
    shortname: str = uc.detector_name_short(longname)
    data, _ = wu.calib_constants(
        shortname,
        exp=exp,
        ctype="geometry",
        dtype="str",
        url=cc.URL,
    )
    if data is None:
        raise ValueError(f"Geometry for {detname} not found in calibration database for experiment {exp} and run {run}.")

    in_file = os.path.join(os.getcwd(), "temp.data")
    with open(in_file, "w") as f:
        f.write(data)

    return in_file


def push(exp: str, run: int, detname: str, out_file: str,) -> None:
    """
    Upload the geometry to the calibration database.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detname : str
        Name of the detector
    out_file : str
        Path to the geometry file to be uploaded
    """
    ctype = "geometry"
    dtype = "str"
    data = mu.data_from_file(out_file, ctype, dtype)
    ds = DataSource(exp=exp, run=run, skip_calib_load="all", max_events=1)
    runs = next(ds.runs())
    detector = runs.Detector(detname)
    longname: str = detector.raw._uniqueid
    shortname: str = uc.detector_name_short(longname)
    det_type: str = detector._dettype
    run_orig: int = run
    run_beg: int = run
    run_end: str = "end"
    kwa = {
        "iofname": out_file,
        "experiment": exp,
        "ctype": ctype,
        "dtype": dtype,
        "detector": shortname,
        "shortname": shortname,
        "detname": detname,
        "longname": longname,
        "run": run,
        "run_beg": run_beg,
        "run_end": run_end,
        "run_orig": run_orig,
        "dettype": det_type,
    }
    _ = wu.deploy_constants(
        data,
        exp,
        longname,
        url=cc.URL_KRB,
        krbheaders=cc.KRBHEADERS,
        **kwa,
    )


def check(exp: str, run: int, detname: str) -> str:
    """
    Check for the geometry file for a given experiment, run, and detector name in the experiment calibration directory.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    detname : str
        Name of the detector
    
    Returns
    -------
    in_file : str
        Path to the geometry file if found, otherwise raises FileNotFoundError
    """
    cdir = f"/sdf/data/lcls/ds/{exp[:3]}/{exp}/calib"
    src = source_from_detname(detname, exp[:3])
    group = group_from_detname(detname)
    ctype = "geometry"
    calib_dir = os.path.join(cdir, src, group, ctype)
    in_file = select_calib_file(calib_dir, run)
    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Geometry for {detname} not found in experiment calibration directory for experiment {exp} and run {run}.")

    return in_file


def fetch_template(detname, pixel_size=None, shape=None):
    """
    Get the template geometry file for a given detector name, and update it with the provided pixel size and shape if it is a Rayonix detector.

    Parameters
    ----------
    detname : str
        Detector name
    pixel_size : float
        Pixel size in µm
    shape : tuple
        Detector shape
    """
    current_dir = os.path.dirname(__file__)
    in_file = os.path.join(current_dir, "templates", f"geometry-def-{detname}.data")

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Template not found for detector {detname}.")
    
    with open(in_file, "r") as file:
        content = file.readlines()

    if detname.lower() == "rayonix":
        if pixel_size is None or shape is None:
            raise ValueError("Pixel size and shape must be provided for Rayonix detector.")

        for i, line in enumerate(content):
            if "MTRX:V2" in line:
                updated_line = re.sub(
                    r"MTRX:V2:\d+:\d+:\d+:\d+",
                    f"MTRX:V2:{shape[0]}:{shape[1]}:{pixel_size}:{pixel_size}",
                    line
                )
                content[i] = updated_line
                break

        with open(in_file, "w") as file:
            file.writelines(content)
    return in_file