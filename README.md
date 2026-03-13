# LCLSGeom
LCLSwapping Geometry tool - helper classes and functions for managing and converting geometry files for LCLS-I and LCLS-II detectors.

# Package Description

This repository aims at providing an easy-to-use geometry management and conversion tool for LCLS scripts. The source code mainly relies on [PSCalib](https://github.com/lcls-psana/PSCalib/tree/master) for LCLS-I detectors and [psana](https://github.com/slac-lcls/lcls2/tree/master) for LCLS-II detectors, both repositeries designed by Mikhail Dubrovin.

## Understanding Geometry Framework

I'll try to go very briefly on explaining the idea behind every geometry format, aiming to highlight the motivation for this project.

### Psana & CrystFEL

Both psana and CrystFEL geometry formats describe detectors using a hierarchical 3D structure. In this representation, the detector geometry is defined through multiple coordinate frames arranged in a hierarchy:
```
Interaction Point (IP)
        ↓
  Detector Center
        ↓
   Panels / ASICs
```

Each level defines the position and orientation of its children relative to its own coordinate frame. Individual panels (or ASICs) are therefore described by their translation and rotation with respect to the detector center, which itself is positioned relative to the interaction point.

This approach is particularly well suited for modern pixel detectors used at LCLS, which are typically segmented detectors composed of multiple panels arranged in 3D space. The hierarchical structure allows accurate modeling of panel offsets, tilts, and metrology corrections.

For example, here is the structure of a `psana` .data file:

```
# TITLE      Geometry parameters of ePix10k2M
# DATE_TIME  2025-01-01T00:00:00 PST
# METROLOGY  no metrology available
# AUTHOR     dubrovin
# EXPERIMENT any
# DETECTOR   epix10k2M
# CALIB_TYPE geometry
# COMMENT:01 Manually created as an example for the 16-segment ePix10k2M detector

# PARAM:01 PARENT     - name and version of the parent object
# PARAM:02 PARENT_IND - index of the parent object
# PARAM:03 OBJECT     - name and version of the object
# PARAM:04 OBJECT_IND - index of the new object
# PARAM:05 X0         - x-coordinate [um] of the object origin in the parent frame
# PARAM:06 Y0         - y-coordinate [um] of the object origin in the parent frame
# PARAM:07 Z0         - z-coordinate [um] of the object origin in the parent frame
# PARAM:08 ROT_Z      - object design rotation angle [deg] around Z axis of the parent frame
# PARAM:09 ROT_Y      - object design rotation angle [deg] around Y axis of the parent frame
# PARAM:10 ROT_X      - object design rotation angle [deg] around X axis of the parent frame
# PARAM:11 TILT_Z     - object tilt angle [deg] around Z axis of the parent frame
# PARAM:12 TILT_Y     - object tilt angle [deg] around Y axis of the parent frame
# PARAM:13 TILT_X     - object tilt angle [deg] around X axis of the parent frame

# HDR PARENT IND     OBJECT IND    X0[um]   Y0[um]   Z0[um]   ROT-Z  ROT-Y  ROT-X     TILT-Z    TILT-Y    TILT-X

      CAMERA  0  EPIX10KA:V1  0    -58583    24251        5     270      0      0   -1.19826   0.11768  -0.03984 # This line sets where is Panel 0 relative to CAMERA
      CAMERA  0  EPIX10KA:V1  1    -57864    64862       66     270      0      0   -1.16980  -0.22605   0.07420 # ...
      CAMERA  0  EPIX10KA:V1  2    -16705    23716      -87     270      0      0   -0.88179  -0.04979  -0.06790 # ...
      CAMERA  0  EPIX10KA:V1  3    -16091    64293        1     270      0      0   -1.12086  -0.09023  -0.10545 # ...
      CAMERA  0  EPIX10KA:V1  4     24315    58774       -4     180      0      0   -0.94978  -0.01083  -0.15229 # ...
      CAMERA  0  EPIX10KA:V1  5     64818    58098       -5     180      0      0   -0.94860  -0.02310  -0.14836 # ...
      CAMERA  0  EPIX10KA:V1  6     23737    16960       35     180      0      0   -0.81313   0.10535  -0.22019 # ...
      CAMERA  0  EPIX10KA:V1  7     64176    16289       15     180      0      0   -0.89935  -0.13787   0.18512 # ...
      CAMERA  0  EPIX10KA:V1  8     58761   -24236      113      90      0      0   -1.18462  -0.05487  -0.00937 # This line sets where is Panel i relative to CAMERA
      CAMERA  0  EPIX10KA:V1  9     58014   -65006      -75      90      0      0   -1.09273   0.06857  -0.19293 # ...
      CAMERA  0  EPIX10KA:V1 10     16766   -23772     -150      90      0      0   -0.91635  -0.15069   0.04136 # ...
      CAMERA  0  EPIX10KA:V1 11     16175   -64142       59      90      0      0   -0.92756  -0.14366  -0.04918 # ...
      CAMERA  0  EPIX10KA:V1 12    -24466   -58939       17       0      0      0   -1.24074  -0.00144  -0.03904 # ...
      CAMERA  0  EPIX10KA:V1 13    -65201   -58201       32       0      0      0   -0.90080   0.02960   0.23655 # ...
      CAMERA  0  EPIX10KA:V1 14    -23760   -16774      -49       0      0      0   -0.66263  -0.27791  -0.04138 # ...
      CAMERA  0  EPIX10KA:V1 15    -64091   -16171       24       0      0      0   -1.09707   0.02238  -0.01484 # ...
          IP  0       CAMERA  0         0    -1000   100000      90      0      0    0.00000   0.00000   0.00000 # This line sets where CAMERA is relative to IP
```

As you can notice, this example .data file specifies an offset in the fast dimension of -0.001 m, a detector-sample distance of 0.1 m and that the detector is rotated by 90 degrees around the detector's normal.
For recap, changing rows in data is slower than changing columns when accessing data `data[row][column]`. Hence the slow dimension is the data row dimension and the fast dimension is the data column dimension.

### PyFAI

Most PyFAI workflows, however, assume a single monolithic 2D detector with a regular pixel grid. In this model, pixels are indexed directly in a 2D array and their positions are inferred from the detector geometry parameters (distance, center etc...).

For segmented detectors, this means that raw detector data must first be assembled into a single 2D image before PyFAI can operate on it. Assembly is the process of projecting the individual detector panels from their 3D positions into a continuous 2D pixel layout.

While this approach works, it introduces an additional preprocessing step that must be repeated each time an image is processed, which can become computationally inefficient when dealing with large datasets. Furthermore, the assembly step implicitly projects the detector geometry onto a single plane. For detectors that are not perfectly planar, this projection forces a non-planar geometry into a planar representation, which can introduce small geometric distortions and artefacts in the reconstructed image and potentially propagate into downstream analysis.

### LCLSGeom Motivation

LCLSGeom aims to bridge the gap between these two geometry paradigms.

Instead of assembling the detector image at runtime, LCLSGeom constructs a virtual 2D detector representation compatible with PyFAI. This is achieved by creating a fake 2D detector where the panel dimensions are expanded along the slow-scan direction so that all panel pixels can be mapped into a single continuous 2D indexing scheme.

The true physical pixel positions are then encoded through the detector metrology information, which is passed to PyFAI. This allows PyFAI to correctly interpret the spatial layout of the pixels while still operating on a single 2D detector structure.

The metrology information can originate from any .data geometry file. As a result, *the final 2D detector geometry depends on the chosen metrology*. If the input .data file presents axis offsets (such as a detector position which is not centered on the IP for example), the PyFAI PONI parameters will then be defined relative to those offsets.

By encoding the detector layout in this way, LCLSGeom allows PyFAI to operate directly on detector data without requiring explicit image assembly, improving efficiency while preserving accurate detector geometry.

## User Guide

LCLSGeom consists in two modules:
- _manager_: Handles relations with calibration databases (for LCLS-II detectors) or directories (for LCLS-I detectors).
- _converter_: Provides quick conversions from one format to another. Available format are `psana`, `CrystFEL` and `pyFAI`.

### From Psana Geometry to PyFAI detectors

The _PsanaToPyFAI_ class is able to parse a `psana` geometry file and converts the geometry information in a `pyFAI` detector object that can be used to perform any downstream `pyFAI` task.

#### Example Usage

**Import required modules**
```python
from LCLSGeom.manager import get_geometry
from LCLSGeom.converter import PsanaToPyFAI
```

LCLS-I experiment
```python
exp = mfxx49820
run = 8
detname = "epix10k2M"

geometry_file = get_geometry(detname, exp, run) # <-- Will fetch geometry file stored in calibration directory or on LCLSGeom/templates if not found
epix10k2M_detector = PsanaToPyFAI.convert(geometry_file, detname) # <-- Implicitely `image_frame=True` zero-out offsets to set up coordinates in image frame
x, y, z = epix10k2M_detector.calc_cartesian_positions()
x.mean()
```
Output
```python
0.0
```

LCLS-II experiment
```python
exp = "mfx101591226"
run = 20
detname = "jungfrau"

geometry_file = get_geometry(detname, exp, run) # <-- Will fetch geometry file stored in slac-lcls/lcls2 or on LCLSGeom/templates if not found
jungfrau_detector = PsanaToPyFAI.convert(geometry_file) # <-- Implicitely `image_frame=True` zero-out offsets to set up coordinates in image frame
x, y, z = jungfrau_detector.calc_cartesian_positions()
x.mean()
```
Output
```python
0.0
```

#### Nota Bene

By default, PsanaToPyFAI.convert removes any axis offsets present in the metrology .data file. This ensures that the resulting PyFAI geometry is centered at (x, y) = (0, 0) and z = 0. In other words, the detector origin is placed at the image center, rather than at the interaction point (IP).
If needed, this behavior can be disabled by passing `image_frame=False`. In that case, the PyFAI geometry is defined in the laboratory coordinate frame.

##### Extra Arguments Example Usage
If we convert the original .data example with an offset in the slow dimension, here is what we can get by passing `image_frame=False`.
 ```python
exp = mfxx49820
run = 8
detname = "epix10k2M"

geometry_file = get_geometry(detname) # <-- Will fetch default geometry file stored in LCLSGeom/templates
epix10k2M_detector = PsanaToPyFAI.convert(geometry_file, detname, image_frame=False) # <-- Do not zero-out offsets to set up coordinates in laboratory frame
x, y, z = jungfrau_detector.calc_cartesian_positions()
x.mean()
```
Output
```python
-0.001
```

### From Psana Geometry to CrystFEL geometry file

The _PsanaToCrystFEL_ class is able to parse a `psana` geometry file and converts the geometry information into a `CrystFEL` geometry file.

#### Example Usage
```python
from LCLSGeom.manager import get_geometry
from LCLSGeom.converter import PsanaToCrystFEL

# LCLS-I experiment
exp = mfxx49820
run = 8
detname = "epix10k2M"

data_file = get_geometry(exp, run, detname)
```

### From PyFAI detectors back to Psana

# Organization

The package structure is organized as follows:

```
LCLSGeom/
├── LCLSGeom/       # Main Package
│ ├── init.py       # Package initialization
│ ├── converter.py  # Conversion Module
│ ├── manager.py    # I/O File Module
│ ├── detector.py   # Detector Definitions
│ ├── geometry.py   # Geometric and Trigonometric Utils
│ ├── frame.py      # Frame Transformation Utisl
│ ├── calib.py      # Calibration Path Utils
│ ├── header.py     # Geometry File Header Utils
│ ├── templates/    # Template Files
│ │ ├── geometry-def-epix10k2M.data
│ │ ├── ...
│ │ └── geometry-def-rayonix.data
├── LICENSE
├── README.md
└──  pyproject.toml # Dependencies and requirements
```