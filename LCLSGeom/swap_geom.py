import os
import sys
import numpy as np
from math import atan2, degrees, sqrt
from pyFAI.detectors import Detector
from PSCalib.UtilsConvert import header_crystfel, panel_constants_to_crystfel
from PSCalib.GeometryAccess import GeometryAccess
from PSCalib.SegGeometryStore import sgs
import PSCalib.GlobalUtils as gu

DETTYPE_TO_PARS = {
    'epix10k2m': ('EPIX10KA:V2','p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,'\
                            'p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0'),
    'epix10kaquad': ('EPIX10KA:V2','p0a0,p1a0,p2a0,p3a0'),
    'jungfrau1m': ('JUNGFRAU:V2','p0a0,p1a0'),
    'jungfrau4m': ('JUNGFRAU:V2','p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0'),
    'rayonix': ('MTRX:V2:1920:1920:89:89','p0a0'),
    'cspad'   : ('SENS2X1:V1', 'p0a0,p0a2,p0a4,p0a6,p0a8,p0a10,p0a12,p0a14,'\
                            'p1a0,p1a2,p1a4,p1a6,p1a8,p1a10,p1a12,p1a14,'\
                            'p2a0,p2a2,p2a4,p2a6,p2a8,p2a10,p2a12,p2a14,'\
                            'p3a0,p3a2,p3a4,p3a6,p3a8,p3a10,p3a12,p3a14'),\
    'cspadv2' : ('SENS2X1:V1', 'p0a0,p1a0,p2a0,p3a0,p4a0,p5a0,p6a0,p7a0,'\
                            'p8a0,p9a0,p10a0,p11a0,p12a0,p13a0,p14a0,p15a0,'\
                            'p16a0,p17a0,p18a0,p19a0,p20a0,p21a0,p22a0,p23a0,'\
                            'p24a0,p25a0,p26a0,p27a0,p28a0,p29a0,p30a0,p31a0'),\
    'pnccd'   : ('MTRX:V2:512:512:75:75', 'p0a0,p1a0,p2a0,p3a0'),\
}

class ePix10k2M(Detector):
    """
    PyFAI Detector instance for the ePix10k2M
    """

    def __init__(
        self,
        pixel1=0.0001,
        pixel2=0.0001,
        n_modules=16,
        n_asics=4,
        asics_shape = (2, 2), # (rows, cols) = (ss, fs)
        fs_size=192,
        ss_size=176,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[0] * ss_size, asics_shape[1] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class ePix10kaQuad(Detector):
    """
    PyFAI Detector instance for the ePix100
    """

    def __init__(
        self,
        pixel1=0.0001,
        pixel2=0.0001,
        n_modules=4,
        n_asics=4,
        asics_shape = (2, 2),
        fs_size=192,
        ss_size=176,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[0] * ss_size, asics_shape[1] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Jungfrau1M(Detector):
    """
    PyFAI Detector instance for the Jungfrau1M
    """

    def __init__(
        self,
        pixel1=0.000075,
        pixel2=0.000075,
        n_modules=2,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        fs_size=256,
        ss_size=256,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[0] * ss_size, asics_shape[1] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Jungfrau4M(Detector):
    """
    PyFAI Detector instance for the Jungfrau4M
    """

    def __init__(
        self,
        pixel1=0.000075,
        pixel2=0.000075,
        n_modules=8,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        fs_size=256,
        ss_size=256,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * asics_shape[0] * ss_size, asics_shape[1] * fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class Rayonix(Detector):
    """
    PyFAI Detector instance for the Rayonix
    """

    def __init__(
        self,
        pixel1=0.000088,
        pixel2=0.000088,
        n_modules=1,
        n_asics=1,
        asics_shape=(1, 1),
        fs_size=1920,
        ss_size=1920,
        **kwargs,
    ):
        super().__init__(pixel1=pixel1, pixel2=pixel2, max_shape=(n_modules * ss_size, fs_size), **kwargs)
        self.n_modules = n_modules
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = ss_size
        self.fs_size = fs_size
        self.pixel_size = pixel1

class PsanatoCrystFEL:
    """
    Class to convert psana .data geometry files to CrystFEL .geom geometry files in the desired reference frame
    """

    def __init__(self, psana_file, output_file, cframe=gu.CFRAME_PSANA, zcorr_um=None):
        self.geometry_to_crystfel(psana_file, output_file, cframe, zcorr_um)

    def geometry_to_crystfel(self, psana_file, output_file, cframe=gu.CFRAME_PSANA, zcorr_um=None):
        geo = GeometryAccess(path=psana_file, pbits=0, use_wide_pix_center=False)
        x, y, z = geo.get_pixel_coords(oname=None, oindex=0, do_tilt=True, cframe=cframe)
        geo1 = geo.get_seg_geo() # GeometryObject
        seg = geo1.algo # object of the SegmentGeometry subclass
        nsegs = int(x.size/seg.size())
        shape = (nsegs,) + seg.shape() # (nsegs, srows, scols)
        x.shape = shape
        y.shape = shape
        z.shape = shape
        txt = header_crystfel()
        for n in range(nsegs):
            z_um = z[n,:]
            if zcorr_um is not None: z_um -= zcorr_um
            txt += panel_constants_to_crystfel(seg, n, x[n,:], y[n,:], z_um)
        if output_file is not None:
            f = open(output_file,'w')
            f.write(txt)
            f.close()

class CrystFELtoPyFAI:
    """
    Class to convert CrystFEL .geom geometry files from a given reference frame to PyFAI corner arrays
    """

    def __init__(self, geom_file, det_type, psana_file=None, cframe=gu.CFRAME_PSANA):
        self.detector = self.get_detector(det_type)
        self.panels = self.from_CrystFEL(geom_file)
        self.pix_pos = self.get_pixel_coordinates(self.panels, psana_file)
        self.corner_array = self.get_corner_array(self.pix_pos, self.panels, cframe)
        self.detector.set_pixel_corners(self.corner_array)

    def get_detector(self, det_type):
        """
        Instantiate a PyFAI Detector object based on the detector type

        Parameters
        ----------
        det_type : str
            Detector type
        """
        if det_type == "epix10k2M":
            return ePix10k2M()
        elif "Epix10kaQuad" in det_type:
            return ePix10kaQuad()
        elif det_type == "jungfrau1M":
            return Jungfrau1M()
        elif det_type == "jungfrau4M":
            return Jungfrau4M()
        elif det_type == "Rayonix":
            return Rayonix()
        else:
            raise ValueError("Detector type not recognized")

    def from_CrystFEL(self, fname: str):
        """
        Parse a CrystFEL geometry file
        Read a text ".geom" file and return the dictionary of geometry components

        Parameters
        ----------
        fname : str
            Path to the CrystFEL geometry file
        """
        detector = {
            "panels": {},
            "rigid_groups": {},
            "rigid_group_collections": {},
        }
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                # Remove comments
                if line[0] == ";":
                    continue
                if "=" not in line:
                    continue
                fmt_line = line.strip()
                # CrystFEL fmt: object = val
                obj_value = line.split("=")
                # object fmt: "panel/parameter"
                obj = obj_value[0].split("/")  # May be len 1 or 2
                value = obj_value[1]
                if len(obj) == 1:  # e.g. rigid_group_quad ...
                    if "collection" in obj[0].strip():
                        collection_name = obj[0].strip().split("_")[-1]
                        detector["rigid_group_collections"][
                            collection_name
                        ] = value.strip().split(",")
                    else:
                        group_name = obj[0].strip().split("_")[-1]
                        detector["rigid_groups"][group_name] = value.strip().split(",")
                elif len(obj) == 2:  # e.g. p0a0/fs = ...
                    pname = obj[0].strip()
                    if pname in detector["panels"]:
                        panel = detector["panels"][pname]
                    else:
                        panel = {
                            "fs": (0, 0, 0),
                            "ss": (0, 0, 0),
                            "res": 10000,
                            "corner_x": 100,
                            "corner_y": 100,
                            "coffset": 0.1,
                            "min_fs": 0,
                            "max_fs": 123,
                            "min_ss": 0,
                            "max_ss": 123,
                            "no_index": 0,
                        }
                        detector["panels"][pname] = panel
                    if "fs" in obj[1].strip()[-2:]:
                        if "max" in obj[1]:
                            panel["max_fs"] = int(value)
                        elif "min" in obj[1]:
                            panel["min_fs"] = int(value)
                        else:
                            strcoords = value.split()
                            if "z" in strcoords:
                                # -1x -2y -3z
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    float(strcoords[2].strip("z")),
                                )
                                panel["fs"] = fcoords
                            else:
                                # -1x -2y
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    0.0,
                                )
                                panel["fs"] = fcoords
                    elif "ss" in obj[1].strip()[-2:]:
                        if "max" in obj[1]:
                            panel["max_ss"] = int(value)
                        elif "min" in obj[1]:
                            panel["min_ss"] = int(value)
                        else:
                            strcoords = value.split()
                            if "z" in strcoords:
                                # -1x -2y -3z
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    float(strcoords[2].strip("z")),
                                )
                                panel["ss"] = fcoords
                            else:
                                # -1x -2y
                                fcoords = (
                                    float(strcoords[0].strip("x")),
                                    float(strcoords[1].strip("y")),
                                    0.0,
                                )
                                panel["ss"] = fcoords
                    elif "res" in obj[1].strip():
                        panel["res"] = float(value)
                    elif "corner" in obj[1].strip():
                        if "x" in obj[1]:
                            panel["corner_x"] = float(value)
                        elif "y" in obj[1]:
                            panel["corner_y"] = float(value)
                    elif "no_index" in obj[1]:
                        panel["no_index"] = int(value)
                    elif "coffset" in obj[1]:
                        panel["coffset"] = float(value)
            return detector

    def get_pixel_coordinates(self, panels: dict, psana_file):
        """
        From either a CrystFEL .geom file or a psana .data file, return the pixel positions

        Parameters
        ----------
        panels : dict
            Dictionary of panels from a CrystFEL geometry file
        geom_file : str
            Path to the geometry file
        """
        nmods = self.detector.n_modules
        nasics = self.detector.n_asics
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        pix_arr = np.zeros([nmods, ss_size * asics_shape[0], fs_size * asics_shape[1], 3])
        if psana_file is None:
            for p in range(nmods):
                pname = f"p{p}"
                for asic in range(nasics):
                    asicname = f"a{asic}"
                    full_name = pname + asicname
                    if nasics == 1:
                        arow = 0
                        acol = 0
                    else:
                        arow = asic // (nasics//2)
                        acol = asic % (nasics//2)
                    ss_portion = slice(arow * ss_size, (arow + 1) * ss_size)
                    fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                    res = panels["panels"][full_name]["res"]
                    corner_x = panels["panels"][full_name]["corner_x"] / res
                    corner_y = panels["panels"][full_name]["corner_y"] / res
                    corner_z = panels["panels"][full_name]["coffset"]
                    # Get tile vectors for ss and fs directions
                    ssx, ssy, ssz = np.array(panels["panels"][full_name]["ss"]) / res
                    fsx, fsy, fsz = np.array(panels["panels"][full_name]["fs"]) / res
                    coords_ss, coords_fs = np.meshgrid(
                        np.arange(0, ss_size), np.arange(0, fs_size), indexing="ij"
                    )
                    x = corner_x + ssx * coords_ss + fsx * coords_fs
                    y = corner_y + ssy * coords_ss + fsy * coords_fs
                    z = corner_z + ssz * coords_ss + fsz * coords_fs
                    pix_arr[p, ss_portion, fs_portion, 0] = x
                    pix_arr[p, ss_portion, fs_portion, 1] = y
                    pix_arr[p, ss_portion, fs_portion, 2] = z
            if len(np.unique(pix_arr[:, :, :, 2]))==1:
                pix_arr[:, :, :, 2] = 0
            else:
                pix_arr[:, :, :, 2] -= np.mean(pix_arr[:, :, :, 2])
        else:
            geom = GeometryAccess(path=psana_file, pbits=0, use_wide_pix_center=False)
            top = geom.get_top_geo()
            child = top.get_list_of_children()[0]
            x, y, z = geom.get_pixel_coords(oname=child.oname, oindex=0, do_tilt=True, cframe=gu.CFRAME_PSANA)
            for p in range(nmods):
                for asic in range(nasics):
                    if nasics == 1:
                        arow = 0
                        acol = 0
                    else:
                        arow = asic // (nasics//2)
                        acol = asic % (nasics//2)
                    ss_portion = slice(arow * ss_size, (arow + 1) * ss_size)
                    fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                    pix_arr[p, ss_portion, fs_portion, 0] = x[p, ss_portion, fs_portion]
                    pix_arr[p, ss_portion, fs_portion, 1] = y[p, ss_portion, fs_portion]
                    pix_arr[p, ss_portion, fs_portion, 2] = z[p, ss_portion, fs_portion]
            pix_arr[:, :, :, 2] -= np.mean(pix_arr[:, :, :, 2])
            pix_arr /= 1e6
        return pix_arr

    def get_corner_array(self, pix_pos, panels, cframe=gu.CFRAME_PSANA):
        """
        Convert to the corner array needed by PyFAI

        Parameters
        ----------
        pix_pos : np.ndarray
            Pixel positions in .geom reference frame

        panels : dict
            Dictionary of panels from a CrystFEL geometry file

        reference_frame : bool
            If True, convert from CrystFEL reference frame to PyFAI reference frame
            If False, convert from psana reference frame to PyFAI reference frame
        """
        nmods = self.detector.n_modules
        nasics = self.detector.n_asics
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        pixcorner = pix_pos.reshape(nmods * ss_size * asics_shape[0], fs_size * asics_shape[1], 3)
        cx, cy, cz = np.moveaxis(pixcorner, -1, 0)
        # Flattened ss dim, fs, Num corners, ZYX coord
        pyfai_fmt = np.zeros([nmods * ss_size * asics_shape[0], fs_size * asics_shape[1], 4, 3])
        for p in range(nmods):
            pname = f"p{p}"
            for asic in range(nasics):
                full_name = f"{pname}a{asic}"
                if nasics == 1:
                    arow = 0
                    acol = 0
                else:
                    arow = asic // (nasics//2)
                    acol = asic % (nasics//2)
                slab_offset = p * asics_shape[0] *ss_size
                ss_portion = slice(
                    arow * ss_size + slab_offset, (arow + 1) * ss_size + slab_offset
                )
                fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                # Get tile vectors for ss and fs directions
                res = panels["panels"][full_name]["res"]
                ssx, ssy, ssz = np.array(panels["panels"][full_name]["ss"]) / res
                fsx, fsy, fsz = np.array(panels["panels"][full_name]["fs"]) / res
                c1x = cx[ss_portion, fs_portion]
                c1y = cy[ss_portion, fs_portion]
                c1z = cz[ss_portion, fs_portion]
                ss_units = np.array([0, 1, 1, 0])
                fs_units = np.array([0, 0, 1, 1])
                x = c1x[:, :, np.newaxis] + ss_units * ssx + fs_units * fsx
                y = c1y[:, :, np.newaxis] + ss_units * ssy + fs_units * fsy
                z = c1z[:, :, np.newaxis] + ss_units * ssz + fs_units * fsz
                # Convert to PyFAI format for detector definition
                # 0 = z along beam, 1 = dim1 (Y) fs, 2 = dim2 (X) ss
                if cframe==0:
                    # psana frame to pyFAI frame
                    # 0 = z along beam, 1 = dim1 (vertical) fs, 2 = dim2 (horizontal) ss
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = z
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = x
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = y
                elif cframe==1:
                    # Lab frame to pyFAI frame
                    # 0 = z along beam, 1 = dim1 (vertical) fs, 2 = dim2 (horizontal) ss
                    pyfai_fmt[ss_portion, fs_portion, :, 0] = z
                    pyfai_fmt[ss_portion, fs_portion, :, 1] = y
                    pyfai_fmt[ss_portion, fs_portion, :, 2] = x
        return pyfai_fmt

class PyFAItoCrystFEL:
    """
    Class to write CrystFEL .geom geometry files from PyFAI SingleGeometry instance
    """

    def __init__(self, sg, pixel_array, psana_file, output_file):
        self.sg = sg
        self.pixel_array = pixel_array
        self.detector = sg.detector
        self.X, self.Y, self.Z = pixel_array[:, :, :, 0], pixel_array[:, :, :, 1], pixel_array[:, :, :, 2]
        self.correct_geom()
        self.geometry_to_crystfel(psana_file, output_file)

    def rotation(self, X, Y, Z, angle):
        """
        Return the X, Y, Z coordinates rotated by angle

        Parameters
        ----------
        X : np.ndarray
            X coordinates
        Y : np.ndarray
            Y coordinates
        Z : np.ndarray
            Z coordinates
        angle : float 
            rotation angle in radians
        """
        Xr = X * np.cos(angle) - Y * np.sin(angle)
        Yr = X * np.sin(angle) + Y * np.cos(angle)
        return Xr, Yr, Z
    
    def translation(self, X, Y, Z, dx, dy, dz):
        """
        Return the X, Y, Z coordinates translated by dx, dy, dz

        Parameters
        ----------
        X : np.ndarray
            X coordinates
        Y : np.ndarray
            Y coordinates
        Z : np.ndarray
            Z coordinates
        dx : float
            Translation in X in meters
        dy : float
            Translation in Y in meters
        dz : float
            Translation in Z in meters
        """
        X += dx
        Y += dy
        Z += dz
        return X, Y, Z
    
    def PONI_to_center(self, dist=0.1, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0):
        """
        Relate the Point of Normal Incidence (PONI) poni1, poni2, dist to the center of the beam Xc, Yc, Zc

        Parameters
        ----------
        dist : float
            Distance in meters
        poni1 : float
            PONI coordinate in the fast scan dimension in meters
        poni2 : float
            PONI coordinate in the slow scan dimension in meters
        rot1 : float
            Rotation angle around the fast scan axis in radians
        rot2 : float
            Rotation angle around the slow scan axis in radians
        rot3 : float
            Rotation angle around the beam axis in radians
        """
        Xc = poni1+dist*(np.tan(rot2)/np.cos(rot1))
        Yc = poni2-dist*(np.tan(rot1))
        Zc = dist/(np.cos(rot1)*np.cos(rot2))
        return Xc, Yc, Zc
    
    def scale_to_µm(self, x, y, z):
        """
        Scale from meter m to micrometer µm

        Parameters
        ----------
        x : np.ndarray
            x coordinate in meters
        y : np.ndarray
            y coordinate in meters
        z : np.ndarray
            z coordinate in meters
        """
        return x*1e6, y*1e6, z*1e6
    
    def correct_geom(self, dist=0, poni1=0, poni2=0, rot1=0, rot2=0, rot3=0):
        """
        Correct the geometry based on the given parameters found by PyFAI calibration
        Finally scale to micrometers (needed for writing CrystFEL .geom files)

        Parameters
        ----------
        dist : float
            Distance in meters
        poni1 : float
            PONI coordinate in the fast scan dimension in meters
        poni2 : float
            PONI coordinate in the slow scan dimension in meters
        rot1 : float
            Rotation angle around the fast scan axis in radians
        rot2 : float
            Rotation angle around the slow scan axis in radians
        rot3 : float
            Rotation angle around the beam axis in radians
        """
        X, Y, Z = self.X, self.Y, self.Z
        if dist==0:
            dist = self.sg.geometry_refinement.param[0]
            poni1 = self.sg.geometry_refinement.param[1]
            poni2 = self.sg.geometry_refinement.param[2]
            rot1 = self.sg.geometry_refinement.param[3]
            rot2 = self.sg.geometry_refinement.param[4]
            rot3 = self.sg.geometry_refinement.param[5]
        Xc, Yc, Zc = self.PONI_to_center(dist, poni1, poni2, rot1, rot2, rot3)
        X, Y, Z = self.rotation(Y, Z, X, -rot1)
        X, Y, Z = self.rotation(Z, X, Y, -rot2)
        X, Y, Z = self.rotation(X, Y, Z, rot3)
        X, Y, Z = self.translation(X, Y, Z, -Xc, -Yc, -Zc)
        X, Y, Z = self.scale_to_µm(X, Y, Z)
        self.X = X
        self.Y = Y
        self.Z = Z
    
    def geometry_to_crystfel(self, psana_file, output_file, zcorr_um=None):
        """
        From corrected X, Y, Z coordinates, write a CrystFEL .geom file

        Parameters
        ----------
        output_file : str
            Path to the output .geom file
        zcorr_um : float
            Correction to the Z coordinates in micrometers
        """
        X, Y, Z = self.X, self.Y, self.Z
        geom = GeometryAccess(path=psana_file, pbits=0, use_wide_pix_center=False)
        geom1 = geom.get_seg_geo() # GeometryObject
        seg = geom1.algo # object of the SegmentGeometry subclass
        nsegs = int(X.size/seg.size())
        shape = (nsegs,) + seg.shape() # (nsegs, srows, scols)
        X.shape = shape
        Y.shape = shape
        Z.shape = shape
        txt = header_crystfel()
        for n in range(nsegs):
            z_um = Z[n,:]
            if zcorr_um is not None: z_um -= zcorr_um
            txt += panel_constants_to_crystfel(seg, n, X[n,:], Y[n,:], z_um)

        if output_file is not None:
            f = open(output_file,'w')
            f.write(txt)
            f.close()

class CrystFELtoPsana:
    """
    Class to convert CrystFEL .geom geometry files to psana .data geometry files thanks to det_type information
    """
    def __init__(self, geom_file, det_type, output_file):
        self.geom_file = geom_file
        self.det_type = det_type
        self.output_file = output_file
        self.valid = False
        self.load_crystfel_file()
        self.convert_crystfel_to_geometry()

    @staticmethod
    def str_to_int_or_float(s):
        v = float(s)
        if v%1 == 0: v=int(v)
        return v

    @staticmethod
    def sfields_to_xyz_vector(flds):
        """ 
        converts ['+0.002583x', '-0.999997y', '+0.000000z'] to (0.002583, -0.999997, 0.000000)
        """
        v = (float(flds[0].strip('x')), float(flds[1].strip('y')))
        z = float(flds[2].strip('z')) if len(flds)==3 else 0
        v += (z,)
        return v

    @staticmethod
    def angle_and_tilt(a):
        """
        for angle in range [-180,180] returns nearest design angle and tilt.
        output angle range is shifted to positive [0,360]
        """
        desangles = np.array((-180,-90, 0, 90, 180))
        difangles = a-desangles
        absdifang = np.absolute(difangles)
        imin = np.where(absdifang == np.amin(absdifang))[0]
        angle, tilt = desangles[imin], difangles[imin]
        return (angle if angle>=0 else angle+360), tilt

    @staticmethod
    def unit_vector_pitch_angle_max_ind(u):
        """
        unit vector pitch (axis transverse direction in x-y plane) angle
        """
        absu = np.absolute(u)
        imax = np.where(absu == np.amax(absu))[0]
        pitch = degrees(atan2(u[2],u[imax]))
        pitch = (pitch+180) if pitch<-90 else (pitch-180) if pitch>90 else pitch
        return pitch, imax

    @staticmethod
    def vector_lab_to_psana(v):
        """
        both-way conversion of vectors between LAB and PSANA coordinate frames
        """
        assert len(v)==3
        return np.array((-v[1], -v[0], -v[2]))

    @staticmethod
    def tilt_xy(uf, us, i, k):
        tilt_f, imaxf = CrystFELtoPsana.unit_vector_pitch_angle_max_ind(uf)
        tilt_s, imaxs = CrystFELtoPsana.unit_vector_pitch_angle_max_ind(us)
        tilt_x, tilt_y = (tilt_s, tilt_f) if imaxf==0 else (tilt_f, tilt_s)
        return tilt_x, -tilt_y

    @staticmethod
    def str_is_segment_and_asic(s):
        """ 
        check if s looks like str 'q0a2' or 'p12a7'
        returns 'p0.2' or 'p12.7' or False
        """
        if not isinstance(s, str)\
        or len(s)<2: return False
        flds = s[1:].split('a')
        return False if len(flds) !=2 else\
            'p%sa%s' % (flds[0], flds[1]) if all([f.isdigit() for f in flds]) else\
            False

    @staticmethod
    def header_psana(list_of_cmts=[], dettype='N/A'):
        comments = '\n'.join(['# CFELCMT:%02d %s'%(i,s) for i,s in enumerate(list_of_cmts)])
        return\
        '\n# TITLE      Geometry constants converted from CrystFEL by genuine psana'\
        +'\n# DATE_TIME  %s' % gu.str_tstamp(fmt='%Y-%m-%dT%H:%M:%S %Z')\
        +'\n# AUTHOR     %s' % gu.get_login()\
        +'\n# CWD        %s' % gu.get_cwd()\
        +'\n# HOST       %s' % gu.get_hostname()\
        +'\n# COMMAND    %s' % ' '.join(sys.argv)\
        +'\n# RELEASE    %s' % gu.get_enviroment('CONDA_DEFAULT_ENV')\
        +'\n# CALIB_TYPE geometry'\
        +'\n# DETTYPE    %s' % dettype\
        +'\n# DETECTOR   N/A'\
        '\n# METROLOGY  N/A'\
        '\n# EXPERIMENT N/A'\
        +'\n%s' % comments\
        +'\n#'\
        '\n# HDR PARENT IND        OBJECT IND     X0[um]   Y0[um]   Z0[um]   ROT-Z ROT-Y ROT-X     TILT-Z   TILT-Y   TILT-X'


    def _parse_line_as_parameter(self, line):
        assert isinstance(line, str), 'line is not a str object'

        fields = line.split()
        nfields = len(fields)

        if fields[1] != '=':
            self.list_of_ignored_records.append(line)
            return

        keys = fields[0].split('/') # ex: p15a3/corner_y

        nkeys = len(keys)
        if nkeys==1:
            if nfields>3:
                self.list_of_ignored_records.append(line)
                return
            k0 = keys[0]
            self.dict_of_pars[k0] = float(fields[2]) if k0 in ('res', 'adu_per_eV', 'coffset') else\
                ' '.join(fields[2:])

        elif nkeys==2:
            k0, k1 = keys
            resp = CrystFELtoPsana.str_is_segment_and_asic(k0)
            if resp: k0=resp
            v = '' if nfields<3 else\
                CrystFELtoPsana.sfields_to_xyz_vector(fields[2:]) if k1 in ('fs','ss') else\
                int(fields[2]) if k1 in ('max_ss', 'min_ss', 'max_fs', 'min_fs', 'no_index') else\
                int(fields[2]) if k1 in ('max_x', 'min_x', 'max_y', 'min_y') else\
                float(fields[2]) if k1 in ('res', 'corner_x', 'corner_y', 'adu_per_eV', 'coffset') else\
                float(fields[2]) if k1 in ('xfs', 'yfs', 'xss', 'yss') else\
                ' '.join(fields[2:]) # str_to_int_or_float(fields[2])
            if k0 in self.dict_of_pars.keys():
                self.dict_of_pars[k0][k1] = v
            else:
                self.dict_of_pars[k0] = {k1:v,}

        else:
            self.list_of_ignored_records.append(line)
            return


    def str_list_of_comments(self):
        return 'List of comments\n'\
            + '\n'.join(self.list_of_comments)


    def str_list_of_ignored_records(self):
        return 'List of ignored records\n'\
            + '\n'.join(self.list_of_ignored_records)


    def str_dict_of_pars(self):
        keys = sorted(self.dict_of_pars.keys())
        msg = 'dict of parameters with top keys: %s' % ' '.join(keys)
        for k in keys:
            v = self.dict_of_pars[k]
            if isinstance(v,dict):
                msg += '\n%s: %s' % (k, CrystFELtoPsana.str_is_segment_and_asic(k))
                for k2,v2 in v.items(): msg += '\n    %s: %s' % (k2,v2)
            else: msg += '\n%s: %s' % (k,v)
        return msg

    def load_crystfel_file(self, fname=None):

        if fname is not None: self.geom_file = fname
        assert os.path.exists(self.geom_file), 'geometry file "%s" does not exist' % self.geom_file

        self.valid = False

        self.list_of_comments = []
        self.list_of_ignored_records = []
        self.dict_of_pars = {}

        f=open(self.geom_file,'r')
        for linef in f:
            line = linef.strip('\n')

            if not line.strip(): continue # discard empty strings
            if line[0] == ';':            # accumulate list of comments
                self.list_of_comments.append(line)
                continue

            self._parse_line_as_parameter(line)

        f.close()

        self.valid = True


    def crystfel_to_geometry(self, pars):
        segname, panasics = pars
        sg = sgs.Create(segname=segname, pbits=0, use_wide_pix_center=False)

        X,Y,Z = sg.pixel_coord_array()


        PIX_SIZE_UM = sg.get_pix_size_um()
        M_TO_UM = 1e6
        xc0, yc0, zc0 = X[0,0], Y[0,0], Z[0,0]
        rc0 = sqrt(xc0*xc0+yc0*yc0+zc0*zc0)

        zoffset_m = self.dict_of_pars.get('coffset', 0) # in meters

        recs = CrystFELtoPsana.header_psana(list_of_cmts=self.list_of_comments, dettype=self.det_type)

        segz = np.array([self.dict_of_pars[k].get('coffset', 0) for k in panasics.split(',')])
        meanroundz = round(segz.mean()*1e3)*1e-3 # round z to 1mm
        zoffset_m += meanroundz

        for i,k in enumerate(panasics.split(',')):
            dicasic = self.dict_of_pars[k]
            uf = np.array(dicasic.get('fs', None), dtype=np.float) # unit vector f
            us = np.array(dicasic.get('ss', None), dtype=np.float) # unit vector s
            vf = uf*abs(xc0)
            vs = us*abs(yc0)
            x0pix = dicasic.get('corner_x', 0) # The units are pixel widths of the current panel
            y0pix = dicasic.get('corner_y', 0)
            z0m   = dicasic.get('coffset', 0)
            adu_per_eV = dicasic.get('adu_per_eV', 1)

            v00center = vf + vs
            v00corner = np.array((x0pix*PIX_SIZE_UM, y0pix*PIX_SIZE_UM, (z0m - zoffset_m)*M_TO_UM))
            vcent = v00corner + v00center

            angle_deg = degrees(atan2(uf[1],uf[0]))
            angle_z, tilt_z = CrystFELtoPsana.angle_and_tilt(angle_deg)
            tilt_x, tilt_y = CrystFELtoPsana.tilt_xy(uf,us,i,k)

            recs += '\nDET:VC         0  %12s  %2d' % (segname, i)\
                + '   %8d %8d %8d %7.0f     0     0   %8.5f %8.5f %8.5f'%\
                (vcent[0], vcent[1], vcent[2], angle_z, tilt_z, tilt_y, tilt_x)
        recs += '\nIP             0    DET:VC       0          0        0'\
                ' %8d       0     0     0    0.00000  0.00000  0.00000' % (zoffset_m*M_TO_UM)

        f=open(self.output_file,'w')
        f.write(recs)
        f.close()

    def convert_crystfel_to_geometry(self):
        pars = DETTYPE_TO_PARS.get(self.det_type.lower(), None)
        self.crystfel_to_geometry(pars)