# LCLSGeom
LCLSwapping Geometry tool - miscellaneous functions for converting geometry files for LCLS geometry calibrations and optimizations.

# Package Description

This repository aims at centralizing and uniformizing LCLS routines to write, convert and switch between different geometry files format. The main code mainly relies on the [PSCalib](https://github.com/lcls-psana/PSCalib/tree/master) repository that was designed by Mikhail Dubrovin. For reference frame definition, we will adopt the traditional (x,y,z) description: x being the first coordinate, y the second and z the third.

## Reference Frame Definition & Convention
The geometry of an experiment can be defined using different frames. The first one involved is the _psana_ frame used in metrology: the different translations x, y, z and rotation angles are measured and entered in .data file. Then those metrology data can be converted in Crystel format .geom file. At this point, those data can be chosen to be formatted in the CrystFEL format while being written in the _psana_ system of coordinates or in the _CrystFEL_ system. Finally, pyFAI uses another format and system of coordinates, but we will see that the latter does not matter much.

<ins>Nota Bene</ins>
Every reference frame is defined at the Interaction Point (IP) which is defined as the point where the X-ray beam interacts with the sample. 

### CrystFEL or Laboratory Coordinate System
This is the traditional system of coordinates when conducting an experiment. According to the [confluence documentation](https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry#DetectorGeometry-Coordinateframes) about detector geometry, the _Lab_ frame is defined as such:

+x axis pointing from IP to the north

+y axis pointing from IP to the top

+z axis pointing from IP along the photon beam propagation direction

### Psana Coordinate System
This is the coordinate system used in metrology measurements. According to the [geometry_converter code](https://confluence.slac.stanford.edu/display/PSDM/Geometry+converter+between+psana+and+CrystFEL#GeometryconverterbetweenpsanaandCrystFEL-Generalremarks), the _psana_ coordinate system can be transformed into the _Lab_ coordinate system by doing those operations:

![psana_to_lab](https://github.com/lcls-users/btx/assets/139856363/751893f8-7e11-4506-aa33-721eda35fa60)
Hence, in this system:

+x points towards the ground

+y points from left to right horizontally

+z is the opposite of the beam direction

This coordinate system is convenient for calibrating the geometry since the pixel space in matplotlib definition has the same axis as in the _psana_ system. The top left corner of an image is the origin of the pixel space, the row-space being the vertical axis (i.e. the first dimension of an array) which in psana is the x-coordinate and the column-space being the horizontal axis (i.e. the second dimension of an array) which in psana is the y-coordinate.

### PyFAI Coordinate System
PyFAI's coordinate system is defined as such:

![PONI](https://github.com/lcls-users/btx/assets/139856363/4ab3b2d7-525a-42b2-a2c8-161cdb85c6bb)
Hence, in this system:

+x points towards the top

+y points from left to right horizontally

+z is along the beam direction

## Geometry Format & Environment
Now that the different coordinate systems has been introduced, we will move on explaining how geometries are defined in each geometry file format.

### Psana Format
This is an LCLS internal geometry file format. It uses a hierarchical representation of the detector: first come pixels, then asics, then panels, and then the IP. To be more consice, every children frame is defined in its parent frame. Parents can have multiple children, but every child only has one parent. For example, a geometry definition for the ePix10k2M detector (exp: mfxx49820) will look as such:

![image](https://github.com/LouConreux/LCLSGeom/assets/139856363/2c436fbb-16fd-4fd4-8b45-2a7894da2706)

Every panel _EPIX10KA:V1_ frame origin is expressed in the _CAMERA_ frame by six parameters: X0, Y0, Z0, ROT-Z, ROT-Y, ROT-X (if you want to take into account small angle deviations, the true rotations to be applied should rather be then (ROT-v + TILT-v)). The _CAMERA_ object is the name given to the center of the detector. Likewise, the _CAMERA_ frame origin is then expressed in the IP frame. Then, optimizing the geometry is only a matter of finding the optimal X0, Y0, Z0, ROT-Z, ROT-Y, ROT-X of the _CAMERA_ origin in the IP frame, and further expressed the downstream object frames is the newly defined _CAMERA_ frame.  

For more information, go to [detector geometry definition](https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry#DetectorGeometry-Childgeometryobjectintheparentframe).

### CrystFEL Format

The _CrystFEL_ format is a more straight-forward way of defining a geometry. In a .geom file, there is no hierarchy involved, everything is expressed in the chosen system of coordinates. Let's stick with the previous example, here what the ePix10k2M .geom file will look like:

![image](https://github.com/LouConreux/LCLSGeom/assets/139856363/b4073fc3-22ce-456c-816b-3f11fc24bb8b)


In a .geom file, it assumes that a geometry object is defined by a name _'piaj'_ where _i_ is the panel index and _j_ is the asic index. In the case of the ePix10k2M, _i_ would lie between 0 and 15 (16 panels) and _j_ would lie between 0 and 3 (4 asics per panel). Then, every geometry object 'piaj' has 11 parameters that defines the geometry as a whole:

_fs_: vector in the chosen coordinate system of the fast dimension

_ss_: vector in the chosen coordinate system of the slow dimension

_res_: resolution of the detector (1/pixel size)

_corner_x_: position in x (chosen coordinate system) of the corner of this panel/asic. A corner is defined as the first point in the panel/asic to appear in the image data. Units are in pixel widths of the current panel/asic

_corner_y_: position in y (chosen coordinate system) of the corner of this panel/asic

_coffest_: position in z (chosen coordinate system) of this panel/asic from the position given by the camera length

_min_fs_: minimal pixel index of current panel/asic in fast dimension

_max_fs_: maximal pixel index of current panel/asic in fast dimension

_min_ss_: minimal pixel index of current panel/asic in slow dimension

_max_ss_: maximal pixel index of current panel/asic in slow dimension

_no_index_: do not matter

<ins>Nota Bene</ins>
The reference frame can be chosen here. One can choose to write the pixel corner coordinates and vectors in the _psana_ frame or the _lab_ frame.

### PyFAI Environment

PyFAI takes as an input a flattened pixel corner array (N<sub>mods</sub> x dim(ss), dim(fs), 4, 3) where _N<sub>mods</sub>_ is the number of panels that the detector has, _dim(ss)_ is the number of pixels in the slow-scan direction, _dim(fs)_ the number of pixels in the fast-scan direction, 4 because a pixel has 4 corners, and 3 for the 3 coordinates.

<ins>Nota Bene</ins>
Again, the reference frame can be chosen here. One can choose to write the pixel corner coordinates in the _psana_ frame or the _lab_ frame or _pyFAI_ frame. PyFAI is really versatile, it is able to perform optimization given any frame where the pixel corner coordinates are expressed.

# Motivation: Geometry Optimization Workflow

The final endpoint of this repository is to provide a central framework for dealing and handling geometry files for the calibration of the different LCLS expriments (mfx, mec, cxi...). This will help design a uniform geometry optimization workflow that aims to be applied for the calibration of all the different LCLS detectors (ePix10k2M, Jungfrau4M, ePix10ka...). This workflow will look like this:

![image](https://github.com/LouConreux/LCLSGeom/assets/139856363/93c0672a-3e1d-4065-a21b-1cfa5e5fb9ed)

Let's go into details:

## Geometry Format Conversion

A geometry calibration starts with a _psana_ .data file and is supposed to end with an optimal _psana_ .data file. But pyFAI cannot comprehend this format, hence we need to have uniform scripts that are able to convert any geometry file format into another one, without affecting the geometry data. This will be handled by built-in functions from the [PSCalib](https://github.com/lcls-psana/PSCalib/tree/master) repository (.data to .geom and vice-versa) as well as some hand-coded functions (.geom to pyFAI format and vice-versa).  

## Optimization and Powder Preprocessing

PyFAI optimization will likely to need some preprocessing of the powder (as discussed in [here](https://github.com/lcls-users/btx/issues/375#issuecomment-2156856817)) explaining why I am regrouping these two in one sub-category. 

PyFAI provides a potential robust and fast geometry calibration workflow that was also described in here. PyFAI takes as an input a flattened pixel corner array (N<sub>mods</sub> x dim(ss), dim(fs), 4, 3) where _N<sub>mods</sub>_ is the number of panels that the detector has, _dim(ss)_ is the number of pixels in the slow-scan direction, _dim(fs)_ the number of pixels in the fast-scan direction, 4 because a pixel has 4 corners, and 3 for the 3 coordinates.  This pixel corner array is then used to instantiate a pyFAI Detector object where the optimization can be done given a powder data.

## New Geometry Definition

Once the optimization is done, pyFAI will give 6 optimal parameters:

_dist_: distance IP-PONI where PONI is the Point of Normal Incidence

_poni1_: slow-scan coordinate of PONI

_poni2_: fast-scan coordinate of PONI

_rot1_: rotation over slow-scan direction

_rot2_: rotation over fast-scan direction

_rot3_: rotation over beam direction

![PONI](https://github.com/lcls-users/btx/assets/139856363/4ab3b2d7-525a-42b2-a2c8-161cdb85c6bb)

<ins>Nota Bene</ins>
These parameters in fact define the translations and rotations that need to be applied to correct the current detector frame.

Hence, once finally optimized, the correct geometry is obtained through adequate translations and rotations. Applying those to the uncalibrated X, Y, Z coordinates, we can then write a new _CrystFEL_ .geom file, and finally write a new _psana_ .data file thanks to the PSCalib code.

# Organization

The package structure is organized as follows:

```
LCLSGeom/
├── LCLSGeom/ # Main Package
│ ├── init.py # Package initialization
│ ├── swap_geom.py # Conversion Module
│ ├── geometry.py  # Geometric and Trigonometric Utils
│ ├── detector.py  # Detector Definition
│ ├── calib.py     # Calibration Path Utils
│ ├── templates/   # Template Files
│ │ ├── Rayonix/
│ │ │ ├── 0-end.data
├── LICENSE # License file
├── README.md # Readme file
├── requirements.txt # Dependencies and requirements
└── setup.py # Setup script for packaging
```

# Usage

The main module _swap_geom.py_ consists in the 6 conversion between the 3 discussed framework _Psana_, _CrystFEL_ and _PyFAI_.

## _PsanaToCrystFEL_

This class takes as input a _Psana_ geometry .data file, a path for the output _CrystFEL_ .geom file

### Example of Usage
```
in_file = 'path/to/geom/mfx/mfxx49820/0-end.data'
out_file = 'path/to/geom/mfx/mfxx49820/r0000.geom'
PsanatoCrystFEL(in_file=in_file, out_file=out_file)
```

## _PsanaToPyFAI

This class takes as input a _Psana_ geometry .data file and the _det_type_ for the adequate detector. Additionally, one can specify a custom pixel size and shape for detector that are usually binned (e.g. Rayonix).
At the moment, valid _det_type_ are:
- "epix10k2M"
- "jungfrau1M"
- "jungfrau4M"
- "Rayonix" 
- "Epix10kaQuad.{i}" where i stands for the quad index

### Example of Usage
```
in_file = 'path/to/geom/mfx/mfxx49820/0-end.data'  
det_type = 'epix10k2M'
pixel_size = 0.0001 # 100 microns
shape = (16, 352, 384)
psana_to_pyfai = PsanaToPyFAI(in_file=in_file, det_type=det_type, pixel_size=pixel_size, shape=shape)
detector = psana_to_pyfai.detector
```

## _PyFAIToCrystFEL_

This class takes as input a _Detector_ PyFAI object _detector_, the 6 geometric parameters _params_ where the result of the optimization are stored, any _Psana_ .data file to access the necessary segmentation infos of the detector, and a path for the _CrystFEL_ .geom file _out_file_.

### Example of Usage
```
in_file = 'path/to/geom/mfx/mfxx49820/0-end.data'  
det_type = 'epix10k2M'
pixel_size = 0.0001 # 100 microns
shape = (16, 352, 384)
psana_to_pyfai = PsanaToPyFAI(in_file=in_file, det_type=det_type, pixel_size=pixel_size, shape=shape)
detector = psana_to_pyfai.detector
...
Optimization
...
params = optimizer.result
geom_file = 'path/to/geom/mfx/mfxx49820/r0008.geom'
PyFAIToCrystFEL(detector=detector, params=params, psana_file=in_file, out_file=geom_file)
```

## _CrystFELToPsana_

This class takes as input a _CrystFEL_ .geom file, a valid _det_type_ and a path for the _output_file_. Additionally, one can specify a custom pixel size and shape for detector that are usually binned (e.g. Rayonix).
At the moment, valid _det_type_ are:
- "epix10k2M"
- "jungfrau1M"
- "jungfrau4M"
- "cspad"
- "cspadv2"
- "pnccd"
- "Rayonix
- "Epix10kaQuad.{i}" where i stands for the quad index

### Example of Usage
```
Optimization
...
params = optimizer.result
geom_file = 'path/to/geom/mfx/mfxx49820/r0008.geom'
psana_file = 'path/to/geom/mfx/mfxx49820/8-end.data'
PyFAIToCrystFEL(detector=detector, params=params, psana_file=in_file, out_file=geom_file)
CrystFELToPsana(in_file=geom_file, det_type=det_type, out_file=psana_file, pixel_size=pixel_size, shape=shape)
```

## _PyFAIToPsana_

This class converts _PyFAI_ Detector instance to _Psana_ .data geometry files by using intermediate _CrystFEL_ .geom files 
This class calls successively _PyFAIToCrystFEL_ then _CrystFELToPsana_.
This class takes as input a _Detector_ PyFAI object _detector_, the 6 geometric parameters _params_ where the result of the optimization are stored, any _Psana_ .data file to access the necessary segmentation infos of the detector, and a path for the _Psana_ .data file _out_file_.

## _CrystFELToPyFAI_

This class converts _CrystFEL_ .geom geometry files to _PyFAI_ Detector instance by using intermediate _Psana_ .data files.
This class calls successively _CrystFELToPsana_ then _PsanaToPyFAI_.
This class takes as input a _CrystFEL_ .geom file, a valid _det_type_. Additionally, one can specify a custom pixel size and shape for detector that are usually binned (e.g. Rayonix).
At the moment, valid _det_type_ are:
- "epix10k2M"
- "jungfrau1M"
- "jungfrau4M"
- "cspad"
- "cspadv2"
- "pnccd"
- "Rayonix
- "Epix10kaQuad.{i}" where i stands for the quad index
