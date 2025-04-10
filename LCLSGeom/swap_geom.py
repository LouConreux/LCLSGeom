import os
import sys
import numpy as np
from math import atan2, degrees
import pyFAI
from .detector import get_detector
from .calib import det_type_to_pars
from PSCalib.UtilsConvert import header_crystfel, panel_constants_to_crystfel
from PSCalib.GeometryAccess import GeometryAccess
from PSCalib.SegGeometryStore import sgs
import PSCalib.GlobalUtils as gu

pyFAI.use_opencl = False

class PsanaToCrystFEL:
    """
    Class to convert psana .data geometry files to CrystFEL .geom geometry files in the desired reference frame

    Parameters
    ----------
    in_file : str
        Path to the psana .data file
    out_file : str
        Path to the output CrystFEL .geom file
    cframe : int
        Frame reference to convert to CrystFEL format
        0 = psana frame, 1 = lab frame
    """

    def __init__(self, in_file, out_file, cframe=gu.CFRAME_PSANA):
        self.convert_data_to_geom(in_file=in_file, out_file=out_file, cframe=cframe)

    def convert_data_to_geom(self, in_file, out_file, cframe=gu.CFRAME_PSANA):
        """
        Write a CrystFEL .geom file from a psana .data file using PSCalib.UtilsConvert functions
        """
        geo = GeometryAccess(path=in_file, pbits=0, use_wide_pix_center=False)
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        x, y, z = geo.get_pixel_coords(oname=child.oname, oindex=0, do_tilt=True, cframe=cframe)
        geo1 = geo.get_seg_geo() # GeometryObject
        seg = geo1.algo # object of the SegmentGeometry subclass
        nsegs = int(x.size/seg.size())
        shape = (nsegs,) + seg.shape() # (nsegs, srows, scols)
        x.shape = shape
        y.shape = shape
        z.shape = shape
        txt = header_crystfel()
        for n in range(nsegs):
            txt += panel_constants_to_crystfel(seg, n, x[n,:], y[n,:], z[n,:])
        if out_file is not None:
            f = open(out_file,'w')
            f.write(txt)
            f.close()

class PsanaToPyFAI:
    """
    Class to convert psana .data geometry files directly to PyFAI corner arrays
    bypassing the writing CrystFEL .geom file step

    Parameters
    ----------
    in_file : str
        Path to the psana .data file
    det_type : str
        Detector type
    pixel_size : float
        Pixel size in meters
    shape : tuple
        Detector shape (n_modules, ss_size, fs_size)
    """
    
    def __init__(self, in_file, det_type, pixel_size=None, shape=None):
        self.detector = get_detector(det_type=det_type, pixel_size=pixel_size, shape=shape)
        corner_array = self.get_pixel_corners(in_file=in_file)
        self.detector.set_pixel_corners(ary=corner_array)
    
    def psana_to_pyfai(self, x, y, z):
        """
        Convert psana coordinates to pyfai coordinates
        """
        x = x * 1e-6
        y = y * 1e-6
        z = z * 1e-6
        if len(np.unique(z))==1:
            z = np.zeros_like(z)
        else:
            z -= np.mean(z)
        return -x, y, -z

    def get_pixel_corners(self, in_file):
        geo = GeometryAccess(path=in_file, pbits=0, use_wide_pix_center=False)
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        x, y, z = geo.get_pixel_coords(oname=child.oname, oindex=0, do_tilt=True)
        x, y, z = self.psana_to_pyfai(x, y, z)
        npanels = self.detector.n_modules
        nasics = self.detector.n_asics
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        pixel_corners = np.zeros([npanels * ss_size * asics_shape[0], fs_size * asics_shape[1], 4, 3])
        for p in range(npanels):
            xp = x[p, :]
            yp = y[p, :]
            zp = z[p, :]
            vfs = np.array((\
                xp[0, fs_size * asics_shape[1] - 1] - xp[0, 0],\
                yp[0, fs_size * asics_shape[1] - 1] - yp[0, 0],\
                zp[0, fs_size * asics_shape[1] - 1] - zp[0, 0]))
            vss = np.array((\
                xp[ss_size * asics_shape[0] - 1,0] - xp[0, 0],\
                yp[ss_size * asics_shape[0] - 1,0] - yp[0, 0],\
                zp[ss_size * asics_shape[0] - 1,0] - zp[0, 0]))
            nfs = vfs / np.linalg.norm(vfs)
            nss = vss / np.linalg.norm(vss)
            for a in range(nasics):
                if nasics == 1:
                    arow = 0
                    acol = 0
                else:
                    arow = a // (nasics//2)
                    acol = a % (nasics//2)
                fs_portion = slice(acol * fs_size, (acol + 1) * fs_size)
                ss_portion = slice(arow * ss_size, (arow + 1) * ss_size)
                slab_offset = p * asics_shape[0] * ss_size
                fs_portion_slab = slice(acol * fs_size, (acol + 1) * fs_size)
                ss_portion_slab = slice(arow * ss_size + slab_offset, (arow + 1) * ss_size + slab_offset)
                ssx, ssy, ssz = nss * self.detector.pixel1
                fsx, fsy, fsz = nfs * self.detector.pixel2
                xasic = x[p, ss_portion, fs_portion]
                yasic = y[p, ss_portion, fs_portion]
                zasic = z[p, ss_portion, fs_portion]
                corner_ss = np.array([0, 0, 1, 1])
                corner_fs = np.array([0, 1, 1, 0])
                xasic = xasic[:, :, np.newaxis] + corner_ss * ssx + corner_fs * fsx
                yasic = yasic[:, :, np.newaxis] + corner_ss * ssy + corner_fs * fsy
                zasic = zasic[:, :, np.newaxis] + corner_ss * ssz + corner_fs * fsz
                pixel_corners[ss_portion_slab, fs_portion_slab, :, 0] = zasic
                pixel_corners[ss_portion_slab, fs_portion_slab, :, 1] = xasic
                pixel_corners[ss_portion_slab, fs_portion_slab, :, 2] = yasic
        return pixel_corners

class PyFAIToCrystFEL:
    """
    Class to write CrystFEL .geom geometry files from PyFAI SingleGeometry instance

    Parameters
    ----------
    detector : PyFAI detector instance
        PyFAI detector instance
    params : list
        Detector parameters found by PyFAI calibration
    psana_file : str
        Path to the psana .data file for retrieving segmentation information
    out_file : str
        Path to the output .geom file
    """

    def __init__(self, detector, params, psana_file, out_file):
        self.detector = detector
        self.params = params
        self.correct_geom()
        self.convert_to_geom(psana_file=psana_file, out_file=out_file)

    def rotation_matrix(self, params):
        """
        Compute and return the detector tilts as a single rotation matrix

        Parameters
        ----------
        params : list
            Detector parameters found by PyFAI calibration
        """
        if params is None:
            params = self.params
        cos_rot1 = np.cos(params[3])
        cos_rot2 = np.cos(params[4])
        cos_rot3 = np.cos(params[5])
        sin_rot1 = np.sin(params[3])
        sin_rot2 = np.sin(params[4])
        sin_rot3 = np.sin(params[5])
        # Rotation about axis 1: Note this rotation is left-handed
        rot1 = np.array([[1.0, 0.0, 0.0],
                            [0.0, cos_rot1, sin_rot1],
                            [0.0, -sin_rot1, cos_rot1]])
        # Rotation about axis 2. Note this rotation is left-handed
        rot2 = np.array([[cos_rot2, 0.0, -sin_rot2],
                            [0.0, 1.0, 0.0],
                            [sin_rot2, 0.0, cos_rot2]])
        # Rotation about axis 3: Note this rotation is right-handed
        rot3 = np.array([[cos_rot3, -sin_rot3, 0.0],
                            [sin_rot3, cos_rot3, 0.0],
                            [0.0, 0.0, 1.0]])
        rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)  # 3x3 matrix
        return rotation_matrix
    
    def pyfai_to_psana(self, x, y, z, params):
        """
        Convert back to psana coordinates

        Parameters
        ----------
        x : np.ndarray
            X coordinate in meters
        y : np.ndarray
            Y coordinate in meters
        z : np.ndarray
            Z coordinate in meters
        params : list
            Detector parameters found by PyFAI calibration
        """
        z -= np.mean(z)
        if params is None:
            params = self.params
        cos_rot1 = np.cos(params[3])
        cos_rot2 = np.cos(params[4])
        distance_sample_detector = params[0]*(1/(cos_rot1*cos_rot2))
        z += distance_sample_detector
        x, y, z = x*1e6, y*1e6, z*1e6
        return -x, y, -z

    def correct_geom(self):
        """
        Correct the geometry based on the given parameters found by PyFAI calibration
        Finally scale to micrometers (needed for writing CrystFEL .geom files)
        """
        params = self.params
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        dist = self.params[0]
        poni1 = self.params[1]
        poni2 = self.params[2]
        p1 = p1 - poni1
        p2 = p2 - poni2
        if p3 is None:
            p3 = np.zeros_like(p1) + dist
        else:
            p3 = p3+dist
        coord_det = np.stack((p1, p2, p3), axis=0)
        coord_sample = np.tensordot(self.rotation_matrix(params), coord_det, axes=([1], [0]))
        x, y, z = coord_sample
        x, y, z = self.pyfai_to_psana(x, y, z, params)
        self.X = x
        self.Y = y
        self.Z = z
    
    def convert_to_geom(self, psana_file, out_file):
        """
        From corrected X, Y, Z coordinates, write a CrystFEL .geom file

        Parameters
        ----------
        psana_file : str
            Path to the psana .data file for retrieving segmentation information
        output_file : str
            Path to the output .geom file
        """
        X, Y, Z = self.X, self.Y, self.Z
        geom = GeometryAccess(path=psana_file, pbits=0, use_wide_pix_center=False)
        geom1 = geom.get_seg_geo()
        calib = geom1.oname
        seg = geom1.algo
        shape = self.detector.raw_shape
        X = X.reshape(shape)
        Y = Y.reshape(shape)
        Z = Z.reshape(shape)
        txt = header_crystfel()
        txt += '\n; calib = %s' % calib\
            +'\n'
        for n in range(shape[0]):
            txt += panel_constants_to_crystfel(seg, n, X[n,:], Y[n,:], Z[n,:])
        if out_file is not None:
            f = open(out_file,'w')
            f.write(txt)
            f.close()

class CrystFELToPsana:
    """
    Class to convert CrystFEL .geom geometry files to psana .data geometry files thanks to det_type information

    Parameters
    ----------
    in_file : str
        Path to the CrystFEL .geom file
    det_type : str
        Detector type
    out_file : str
        Path to the output psana .data file
    """
    def __init__(self, in_file, det_type, out_file):
        self.valid = False
        self.load_geom(in_file=in_file)
        self.convert_geom_to_data(det_type=det_type, out_file=out_file)

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
        tilt_f, imaxf = CrystFELToPsana.unit_vector_pitch_angle_max_ind(uf)
        tilt_s, imaxs = CrystFELToPsana.unit_vector_pitch_angle_max_ind(us)
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
    def header_psana(list_of_cmts=[], det_type='N/A', calib='N/A'):
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
        +'\n# DETTYPE    %s' % det_type\
        +'\n# DETECTOR   %s' % calib\
        +'\n# METROLOGY  N/A'\
        '\n# EXPERIMENT N/A'\
        +'\n%s' % comments\
        +'\n'\
        +'\n# HDR PARENT IND     OBJECT IND    X0[um]   Y0[um]   Z0[um]   ROT-Z  ROT-Y  ROT-X     TILT-Z    TILT-Y    TILT-X'\
        '\n'

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
            resp = CrystFELToPsana.str_is_segment_and_asic(k0)
            if resp: k0=resp
            v = '' if nfields<3 else\
                CrystFELToPsana.sfields_to_xyz_vector(fields[2:]) if k1 in ('fs','ss') else\
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
                msg += '\n%s: %s' % (k, CrystFELToPsana.str_is_segment_and_asic(k))
                for k2,v2 in v.items(): msg += '\n    %s: %s' % (k2,v2)
            else: msg += '\n%s: %s' % (k,v)
        return msg

    def load_geom(self, in_file):
        self.valid = False
        self.list_of_comments = []
        self.list_of_ignored_records = []
        self.dict_of_pars = {}
        f=open(in_file,'r')
        for linef in f:
            line = linef.strip('\n')
            if not line.strip(): continue # discard empty strings
            if line[0] == ';':            # accumulate list of comments
                if 'calib' in line:
                    self.calib = line.split('=', 1)[1].strip()
                self.list_of_comments.append(line)
                continue
            self._parse_line_as_parameter(line)
        f.close()
        self.valid = True

    def geom_to_data(self, segname, panelasics, det_type, out_file):
        sg = sgs.Create(segname=segname, pbits=0, use_wide_pix_center=False)
        X, Y, Z = sg.pixel_coord_array()
        PIX_SIZE_UM = sg.get_pix_size_um()
        M_TO_UM = 1e6
        xc0, yc0, zc0 = X[0,0], Y[0,0], Z[0,0]
        zoffset_m = self.dict_of_pars.get('coffset', 0) # in meters
        recs = CrystFELToPsana.header_psana(list_of_cmts=self.list_of_comments, det_type=det_type, calib=segname)
        segz = np.array([self.dict_of_pars[k].get('coffset', 0) for k in panelasics.split(',')])
        meanroundz = round(segz.mean()*1e6)*1e-6 # round z to 1µm
        zoffset_m += meanroundz
        for i,k in enumerate(panelasics.split(',')):
            dicasic = self.dict_of_pars[k]
            uf = np.array(dicasic.get('fs', None), dtype=np.float64) # unit vector f
            us = np.array(dicasic.get('ss', None), dtype=np.float64) # unit vector s
            vf = uf*abs(xc0)
            vs = us*abs(yc0)
            x0pix = dicasic.get('corner_x', 0) # The units are pixel widths of the current panel
            y0pix = dicasic.get('corner_y', 0)
            z0m   = dicasic.get('coffset', 0)
            v00center = vf + vs
            v00corner = np.array((x0pix*PIX_SIZE_UM, y0pix*PIX_SIZE_UM, (z0m - zoffset_m)*M_TO_UM))
            vcent = v00corner + v00center
            angle_deg = degrees(atan2(uf[1],uf[0]))
            angle_z, tilt_z = CrystFELToPsana.angle_and_tilt(angle_deg)
            tilt_x, tilt_y = CrystFELToPsana.tilt_xy(uf,us,i,k)
            recs += '\n      CAMERA  0 %12s %2d' %(segname, i)\
                +'  %8d %8d %8d %7.0f      0      0   %8.5f  %8.5f  %8.5f'%\
                (vcent[0], vcent[1], vcent[2], angle_z, tilt_z, tilt_y, tilt_x)
        recs += '\n          IP  0       CAMERA  0         0        0'\
                ' %8d       0      0      0    0.00000   0.00000   0.00000' % (zoffset_m*M_TO_UM)
        f=open(out_file,'w')
        f.write(recs)
        f.close()

    def convert_geom_to_data(self, det_type, out_file):
        det_type_lower = det_type.lower()
        if "epix10kaquad" in det_type_lower:
            det_type_lower = "epix10kaquad"
            panelasics = det_type_to_pars.get(det_type_lower, None)
        elif det_type_lower == "rayonix":
            panelasics = 'p0a0'
        else:
            panelasics = det_type_to_pars.get(det_type_lower, None)
        self.geom_to_data(self.calib, panelasics, det_type, out_file)

class CrystFELToPyFAI:
    """
    Class to convert CrystFEL .geom geometry files to PyFAI Detector instance by using intermediate psana .data files

    Parameters
    ----------
    in_file : str
        Path to the CrystFEL .geom file
    det_type : str
        Detector type
    pixel_size : float
        Pixel size in meters
    shape : tuple
        Detector shape (n_modules, ss_size, fs_size)
    """
    def __init__(self, in_file, det_type, pixel_size=None, shape=None):
        path = os.path.dirname(in_file)
        data_file = os.path.join(path, "temp.data")
        CrystFELToPsana(in_file=in_file, det_type=det_type, out_file=data_file, pixel_size=pixel_size, shape=shape)
        psana_to_pyfai = PsanaToPyFAI(in_file=data_file, det_type=det_type, pixel_size=pixel_size, shape=shape)
        self.detector = psana_to_pyfai.detector
        os.remove(data_file)

class PyFAIToPsana:
    """
    Class to convert PyFAI Detector instance to psana .data geometry files by using intermediate CrystFEL .geom files 

    Parameters
    ----------
    detector : PyFAI detector instance
        PyFAI detector instance
    params : list
        Detector parameters found by PyFAI calibration
    psana_file : str
        Path to the psana .data file for retrieving segmentation information
    out_file : str
        Path to the output .psana file
    """

    def __init__(self, detector, params, psana_file, out_file):
        path = os.path.dirname(out_file)
        geom_file = os.path.join(path, "temp.geom")
        PyFAIToCrystFEL(detector=detector, params=params, psana_file=psana_file, out_file=geom_file)
        CrystFELToPsana(in_file=geom_file, det_type=detector.det_type, out_file=out_file, pixel_size=detector.pixel_size, shape=detector.raw_shape)
        os.remove(geom_file)