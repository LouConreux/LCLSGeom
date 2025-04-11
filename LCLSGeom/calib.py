import os
import re

det_type_to_pars = {
    'epix10k2m': 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0',
    'epix10kaquad': 'p0a0,p1a0,p2a0,p3a0',
    'jungfrau1m': 'p0a0,p1a0',
    'jungfrau4m': 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0',
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

psana_det_names = (
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

psana_det_names_lower = tuple(name.lower() for name in psana_det_names)

calib_det_names = (
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


psana_to_calib_det_name = dict(zip(psana_det_names_lower, calib_det_names))

det_to_group = dict(zip(calib_det_names, calib_groups))

hutch_to_station = dict(zip(hutches, stations))


def group_from_det_type(det_type: str) -> str:
    """Retrieve the group string from the detector type."""
    det_type_lower = det_type.lower()
    det_name = psana_to_calib_det_name.get(det_type_lower, "UNDEFINED")
    if det_name == "UNDEFINED":
        raise ValueError(f"Unknown detector type: {det_type}")
    group = det_to_group.get(det_name)
    return group


def source_from_det_info(det_type: str, hutch: str) -> str:
    """Retrieve the source string from the detector type and hutch."""
    hutch_upper = hutch.upper()
    station = hutch_to_station.get(hutch_upper, "UNDEFINED")
    if station == "UNDEFINED":
        raise ValueError(f"Unknown hutch: {hutch}")
    det_type_lower = det_type.lower()
    det_name = psana_to_calib_det_name.get(det_type_lower, "UNDEFINED")
    if det_name == "UNDEFINED":
        raise ValueError(f"Unknown detector type: {det_type}")
    if "Epix10kaQuad" in det_name:
        return f"{station}:{det_name}"
    return f"{station}:{det_name}.0"

def fetch_template(exp, det_type, src, pixel_size, shape):
    """
    Pick the appropriate psana format template based on the detector type
    Parameters pixel size, and shape are required for the Rayonix detector
    Populate the experiment calibration directory with the template

    Parameters
    ----------
    exp : str
        Experiment name
    det_type : str
        Detector type
    src : str
        Source name of end station
    pixel_size : float
        Pixel size in µm
    shape : tuple
        Detector shape
    """
    current_dir = os.path.dirname(__file__)
    template_file = os.path.join(current_dir, "templates", det_type, "0-end.data")
    if not os.path.exists(template_file):
        raise FileNotFoundError(f"Template not found for detector {det_type}.")
    
    with open(template_file, "r") as file:
        content = file.readlines()

    if det_type.lower() == "rayonix":
        for i, line in enumerate(content):
            if "MTRX:V2" in line:
                updated_line = re.sub(
                    r"MTRX:V2:\d+:\d+:\d+:\d+",
                    f"MTRX:V2:{shape[0]}:{shape[1]}:{pixel_size}:{pixel_size}",
                    line
                )
                content[i] = updated_line
                break

    cdir = f"/sdf/data/lcls/ds/{exp[:3]}/{exp}/calib"
    group = group_from_det_type(det_type)
    type = "geometry"
    in_file = os.path.join(cdir, group, src, type, "0-end.data")
    os.makedirs(os.path.dirname(in_file), exist_ok=True)
    with open(in_file, "w") as file:
        file.writelines(content)
    return in_file