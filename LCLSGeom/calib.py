"""
This module provides functions to determine the appropriate calibration group and source string for a given LCLS-I detector and hutch.
It also includes mappings between detector names, calibration groups, and hutches/stations.
"""

import os

detname_to_pars = {
    'epix10k2m': 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0',
    'epix10kaquad': 'p0a0,p1a0,p2a0,p3a0',
    'jungfrau05m': 'p0a0',
    'jungfrau1m': 'p0a0,p1a0',
    'jungfrau4m': 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0',
    'jungfrau': 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0',
}

calib_groups = (
    "UNDEFINED",
    "CsPad::CalibV1",
    "CsPad2x2::CalibV1",
    "Princeton::CalibV1",
    "PNCCD::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Epix::CalibV1",
    "Epix10ka::CalibV1",
    "Epix100a::CalibV1",
    "Camera::CalibV1",
    "Andor::CalibV1",
    "Acqiris::CalibV1",
    "Imp::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "EvrData::CalibV1",
    "Camera::CalibV1",
    "Timepix::CalibV1",
    "Fli::CalibV1",
    "Pimax::CalibV1",
    "Andor3d::CalibV1",
    "Jungfrau::CalibV1",
    "Jungfrau::CalibV1",
    "Jungfrau::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Epix10ka::CalibV1",
    "Uxi::CalibV1",
    "Pixis::CalibV1",
    "Epix10ka2M::CalibV1",
    "Epix10kaQuad::CalibV1",
    "Epix10kaQuad::CalibV1",
    "Epix10kaQuad::CalibV1",
    "Epix10kaQuad::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
    "Camera::CalibV1",
)

psana_detnames = (
    "UNDEFINED",
    "Cspad",
    "Cspad2x2",
    "Princeton",
    "pnCCD",
    "Tm6740",
    "Opal1000",
    "Opal2000",
    "Opal4000",
    "Opal8000",
    "OrcaFl40",
    "Epix",
    "Epix10k",
    "Epix100a",
    "Fccd960",
    "Andor",
    "Acqiris",
    "Imp",
    "Quartz4A150",
    "Rayonix",
    "Evr",
    "Fccd",
    "Timepix",
    "Fli",
    "Pimax",
    "Andor3d",
    "Jungfrau",
    "Jungfrau1M",
    "Jungfrau4M",
    "Zyla",
    "ControlsCamera",
    "Epix10ka",
    "Uxi",
    "Pixis",
    "Epix10k2M",
    "Epix10kaQuad0",
    "Epix10kaQuad1",
    "Epix10kaQuad2",
    "Epix10kaQuad3",
    "Streak",
    "Archon",
    "iStar",
    "Alvium",
)

psana_detnames_lower = tuple(name.lower() for name in psana_detnames)

full_detnames = (
    "UNDEFINED",
    "cspad",
    "cspad2x2",
    "princeton",
    "pnCCD",
    "tm6740",
    "opal1000",
    "opal2000",
    "opal4000",
    "opal8000",
    "orcaFl40",
    "epix",
    "epix10k",
    "epix100a",
    "fccd960",
    "andor",
    "acqiris",
    "imp",
    "quartz4A150",
    "rayonix",
    "evr",
    "fccd",
    "timepix",
    "fli",
    "pimax",
    "andor3d",
    "jungfrau16M",
    "jungfrau1M",
    "jungfrau4M",
    "zyla",
    "controlsCamera",
    "epix10ka",
    "uxi",
    "pixis",
    "epix10k2M",
    "epix10kaQuad0",
    "epix10kaQuad1",
    "epix10kaQuad2",
    "epix10kaQuad3",
    "streak",
    "archon",
    "iStar",
    "alvium",
)

calib_detnames = (
    "UNDEFINED",
    "Cspad",
    "Cspad2x2",
    "Princeton",
    "pnCCD",
    "Tm6740",
    "Opal1000",
    "Opal2000",
    "Opal4000",
    "Opal8000",
    "OrcaFl40",
    "Epix",
    "Epix10k",
    "Epix100a",
    "Fccd960",
    "Andor",
    "Acqiris",
    "Imp",
    "Quartz4A150",
    "Rayonix",
    "Evr",
    "Fccd",
    "Timepix",
    "Fli",
    "Pimax",
    "Andor3d",
    "Jungfrau",
    "Jungfrau",
    "Jungfrau",
    "Zyla",
    "ControlsCamera",
    "Epix10ka",
    "Uxi",
    "Pixis",
    "Epix10ka2M",
    "Epix10kaQuad.0",
    "Epix10kaQuad.1",
    "Epix10kaQuad.2",
    "Epix10kaQuad.3",
    "Streak",
    "Archon",
    "iStar",
    "Alvium",
)

hutches = (
    "UNDEFINED",
    "XPP",
    "XCS",
    "CXI",
    "MEC",
    "MFX",
)

stations = (
    "UNDEFINED",
    "XppEndstation.0",
    "XcsEndstation.0",
    "CxiDs1.0",
    "MecTargetChamber.0",
    "MfxEndstation.0",
)


psana_to_calib_detname = dict(zip(psana_detnames_lower, calib_detnames))

psana_to_full_detname = dict(zip(psana_detnames_lower, full_detnames))

det_to_group = dict(zip(calib_detnames, calib_groups))

hutch_to_station = dict(zip(hutches, stations))


def group_from_detname(detname: str) -> str:
    """Retrieve the group string from the detector type."""
    detname_lower = detname.lower()
    detname = psana_to_calib_detname.get(detname_lower, "UNDEFINED")
    if detname == "UNDEFINED":
        raise ValueError(f"Unknown detector type: {detname}")
    group = det_to_group.get(detname)
    return group


def source_from_detname(detname: str, hutch: str) -> str:
    """Retrieve the source string from the detector name and hutch."""
    hutch_upper = hutch.upper()
    station = hutch_to_station.get(hutch_upper, "UNDEFINED")
    if station == "UNDEFINED":
        raise ValueError(f"Unknown hutch: {hutch}")
    detname_lower = detname.lower()
    detname = psana_to_calib_detname.get(detname_lower, "UNDEFINED")
    if detname == "UNDEFINED":
        raise ValueError(f"Unknown detector type: {detname}")
    if "Epix10kaQuad" in detname:
        return f"{station}:{detname}"
    return f"{station}:{detname}.0"


def select_calib_file(calib_dir: str, run: int) -> Optional[str]:
    """Select the calibration file from the calibration directory and run number."""
    fnames = os.listdir(calib_dir)
    files = [os.path.join(calib_dir, fname) for fname in fnames]

    run_max = 9999
    run_files = []
    for file in files:
        f = os.path.basename(file)
        if f == "HISTORY":
            continue
        if os.path.splitext(f)[1] != ".data":
            continue
        basename = os.path.splitext(f)[0]
        fields = basename.split("-")
        begin, end = fields

        if begin.isdigit():
            begin_int = int(begin)
            if begin_int >= run_max:
                raise ValueError(
                    f"Begin run number {run} is too high for calibration directory {calib_dir}"
                )

        if end.isdigit():
            end_int = int(end)
            if end_int >= run_max:
                raise ValueError(
                    f"End run number {run} is too high for calibration directory {calib_dir}"
                )
        elif end == "end":
            end_int = run_max

        run_files.append((begin_int, end_int, file))
    run_files.sort(key=lambda x: int(x[0]))

    for run_file in run_files[::-1]:
        if run_file[0] <= run <= run_file[1]:
            return run_file[2]

    return os.path.join(calib_dir, "0-end.data")


def clean_detname(detname: str) -> str:
    """Clean the detector name by removing any trailing segment and ASIC information."""
    detname_lower = detname.lower()
    full_detname = psana_to_full_detname.get(detname_lower, "UNDEFINED")
    if full_detname == "UNDEFINED":
        raise ValueError(f"Unknown detector type: {detname}")
    return full_detname