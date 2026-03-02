"""
This module provides functions to determine the appropriate calibration group and source string for a given LCLS-I detector and hutch.
It also includes mappings between detector names, calibration groups, and hutches/stations.
"""

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