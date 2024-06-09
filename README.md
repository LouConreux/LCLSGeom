# LCLSGeom
Louis Conreux's Luminescent Swapping Geometry tool - miscellaneous functions for converting geometry files for LCLS geometry calibrations and optimizations.

## Table of Contents

## Motivation

### Geometry Optimization Workflow

## Description

This repository aims at centralizing and uniformizing LCLS routines to write, convert and switch between different geometry files format. The main code mainly relies on the [PSCalib](https://github.com/lcls-psana/PSCalib/tree/master) repository that was designed by Mikhail Dubrovin. For reference frame definition, we will adopt the traditional (x,y,z) description: x being the first coordinate, y the second and z the third.

### Reference Frame Definition & Convention
The geometry of an experiment can be defined using different frames. The first one involved is the _psana_ frame used in metrology: the different translations x, y, z and rotation angles are measured and entered in .data file. Then those metrology data can be converted in Crystel format .geom file. At this point, those data can be chosen to be formatted in the CrystFEL format while being written in the _psana_ system of coordinates or in the _CrystFEL_ system. Finally, pyFAI uses another format and system of coordinates, but we will see that the latter does not matter much.

<ins>Nota Bene</ins>
Every reference frame is defined at the Interaction Point (IP) which is defined as the point where the X-ray beam interacts with the sample. 


#### CrystFEL or Laboratory Coordinate System
This is the traditional system of coordinates when conducting an experiment. According to the [confluence documentation](https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry#DetectorGeometry-Coordinateframes) about detector geometry, the _Lab_ frame is defined as such:

+x axis pointing from IP to the north

+y axis pointing from IP to the top

+z axis pointing from IP along the photon beam propagation direction

#### Psana Coordinate System
This is the coordinate system used in metrology measurements. According to the [geometry_converter code](https://confluence.slac.stanford.edu/display/PSDM/Geometry+converter+between+psana+and+CrystFEL#GeometryconverterbetweenpsanaandCrystFEL-Generalremarks), the _psana_ coordinate system can be transformed into the _CrystFEL_ coordinate system by doing those operations:

![psana_to_lab](https://github.com/lcls-users/btx/assets/139856363/751893f8-7e11-4506-aa33-721eda35fa60)
Hence, in this system:

+x points towards the ground

+y points from left to right horizontally

+z is the opposite of the beam direction

This coordinate system is convenient for calibrating the geometry since the pixel space in matplotlib definition has the same axis as in the _psana_ system. The top left corner of an image is the origin of the pixel space, the row-space being the vertical axis (i.e. the first dimension of an array) which in psana is the x-coordinate and the column-space being the horizontal axis (i.e. the second dimension of an array) which in psana is the y-coordinate.


#### PyFAI Coordinate System
PyFAI's coordinate system is defined as such:

![PONI](https://github.com/lcls-users/btx/assets/139856363/4ab3b2d7-525a-42b2-a2c8-161cdb85c6bb)
Hence, in this system:

+x points towards the top

+y points from left to right horizontally

+z is along the beam direction


#### Frame Transformations
Wrapping everything, the different frame definition can be drawn as such:

![image](https://github.com/LouConreux/LCLSGeom/assets/139856363/10c99409-1c85-463b-be9c-c04b0f692f94)


### Geometry Format & Environment
Now that the different coordinate systems has been introduced, we will move on explaining how geometries are defined in each geometry file format.

#### Psana Format
This is an LCLS internal geometry file format. It uses a hierarchical representation of the detector: first come pixels, then asics, then panels, and then the IP. To be more consice, every children frame is defined in its parent frame. Parents can have multiple children, but every child only has one parent. For example, a geometry definition for ePix10k2M will look as such:
![image](https://github.com/LouConreux/LCLSGeom/assets/139856363/2c436fbb-16fd-4fd4-8b45-2a7894da2706)
Every panel _EPIX10KA:V1_ frame origin is expressed in the _CAMERA_ frame by six parameters: X0, Y0, Z0, ROT-Z, ROT-Y, ROT-X (if you want to take into account small angle deviations, the true rotations to be applied should rather be then (ROT-v + TILT-v)). The _CAMERA_ object is the name given to the center of the detector. Likewise, the _CAMERA_ frame origin is then expressed in the IP frame. Then, optimizing the geometry is only a matter of finding the optimal X0, Y0, Z0, ROT-Z, ROT-Y, ROT-X of the _CAMERA_ origin in the IP frame, and further expressed the downstream object frames is the newly defined _CAMERA_ frame.  

For more information, go to [detector geometry definition](https://confluence.slac.stanford.edu/display/PSDM/Detector+Geometry#DetectorGeometry-Childgeometryobjectintheparentframe).

#### CrystFEL Format


#### PyFAI Environment



