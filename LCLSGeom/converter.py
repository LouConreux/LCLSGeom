import os
import numpy as np
from math import atan2, degrees
import pyFAI
from .detector import get_detector
from .calib import detname_to_pars
from .utils import str_is_segment_and_asic, sfields_to_xyz_vector, header_psana, header_crystfel
from .geometry import rotation_matrix, angle_and_tilt, tilt_xy, rotate_z
import psana
pyFAI.use_opencl = False

class PsanaToCrystFEL:
    """
    Class to convert psana .data geometry files to CrystFEL .geom geometry files in the desired reference frame

    Parameters
    ----------
    exp : str
        Experiment tag
    run_num : int
        Run number
    detname : str
        Detector name
    out_file : str
        Path to the output CrystFEL .geom file
    """

    def __init__(self, exp, run_num, detname, out_file):
        ds = psana.DataSource(exp=exp, run=run_num)
        run = next(ds.runs())
        try:
            self.det = run.Detector(detname)
        except Exception as e:
            raise ValueError(f"Detector {detname} not found in run {run_num} of experiment {exp}. Error: {e}")
        self.convert_data_to_geom(out_file=out_file)

    def convert_data_to_geom(self, out_file):
        """
        Write a CrystFEL .geom file from a psana .data file using PSCalib.UtilsConvert functions

        Parameters
        ----------
        out_file : str
            Path to the output CrystFEL .geom file
        """
        geo = self.det.raw._det_geo()
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        x, y, z = geo.get_pixel_coords(oname=child.oname, oindex=0, do_tilt=True, cframe=0)
        seg = self.det.raw._seg_geo
        shape = self.det.raw._shape_total()
        arows, acols = seg.asic_rows_cols()
        srows, _ = seg.shape()
        pix_size = seg.pixel_scale_size()
        _, nasics_in_cols = seg.number_of_asics_in_rows_cols()
        nasicsf = nasics_in_cols
        x.shape = shape
        y.shape = shape
        z.shape = shape
        txt = header_crystfel()
        for n in range(shape[0]):
            txt += '\n'
            for a,(r0,c0) in enumerate(seg.asic0indices()):
                vfs = np.array((\
                    x[n,r0,c0+acols-1] - x[n,r0,c0],\
                    y[n,r0,c0+acols-1] - y[n,r0,c0],\
                    z[n,r0,c0+acols-1] - z[n,r0,c0]))
                vss = np.array((\
                    x[n,r0+arows-1,c0] - x[n,r0,c0],\
                    y[n,r0+arows-1,c0] - y[n,r0,c0],\
                    z[n,r0+arows-1,c0] - z[n,r0,c0]))
                nfs = vfs/np.linalg.norm(vfs)
                nss = vss/np.linalg.norm(vss)

                pref = '\np%da%d'%(n,a)

                txt += '%s/fs = %+.6fx %+.6fy %+.6fz' % (pref, nfs[0], nfs[1], nfs[2]) \
                    + '%s/ss = %+.6fx %+.6fy %+.6fz' % (pref, nss[0], nss[1], nss[2]) \
                    + '%s/res = %.3f' % (pref, 1e6/pix_size) \
                    + '%s/corner_x = %.6f' % (pref, x[n,r0,c0]/pix_size) \
                    + '%s/corner_y = %.6f' % (pref, y[n,r0,c0]/pix_size) \
                    + '%s/coffset = %.6f' % (pref, z[n,r0,c0]*1e-6) \
                    + '%s/min_fs = %d' % (pref, (a%nasicsf)*acols) \
                    + '%s/max_fs = %d' % (pref, (a%nasicsf+1)*acols-1) \
                    + '%s/min_ss = %d' % (pref, n*srows + (a//nasicsf)*arows) \
                    + '%s/max_ss = %d' % (pref, n*srows + (a//nasicsf+1)*arows - 1) \
                    + '%s/no_index = 0' % (pref) \
                    + '\n'
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
    exp : str
        Experiment tag
    run_num : int
        Run number
    detname : str
        Detector name
    """

    def __init__(self, exp, run_num, detname):
        ds = psana.DataSource(exp=exp, run=run_num)
        run = next(ds.runs())
        try:
            self.det = run.Detector(detname)
        except Exception as e:
            raise ValueError(f"Detector {detname} not found in run {run_num} of experiment {exp}. Error: {e}")
        shape = self.det.raw._shape_total()
        self.detector = get_detector(shape=shape)
        self.setup_detector()
        self.get_pixel_index_map()
        corner_array = self.get_pixel_corners()
        self.detector.set_pixel_corners(ary=corner_array)

    def setup_detector(self):
        """
        Pass the detector segmentation and geometry info to the PyFAI detector instance
        """
        self.detector.geo = self.det.raw._det_geo()
        self.detector.seg = self.det.raw._seg_geo
        self.detector.detname = self.det.raw._det_name()
        self.detector.segname = self.detector.seg.algo.oname

    def psana_to_pyfai(self, x, y, z):
        """
        Convert psana coordinates to pyfai coordinates

        Parameters
        ----------
        x : np.ndarray
            X coordinate in micrometers
        y : np.ndarray
            Y coordinate in micrometers
        z : np.ndarray
            Z coordinate in micrometers
        """
        x = x * 1e-6
        y = y * 1e-6
        z = z * 1e-6
        if len(np.unique(z))==1:
            z = np.zeros_like(z)
        else:
            z -= np.mean(z)
        return -x, y, -z

    def get_pixel_index_map(self):
        """
        Create a pixel index map for assembling the detector
        """
        temp_index = [np.asarray(t) for t in self.detector.geo.get_pixel_coord_indexes()]
        pixel_index_map = np.zeros((np.array(temp_index).shape[2:]) + (2,))
        pixel_index_map[..., 0] = temp_index[0][0]
        pixel_index_map[..., 1] = temp_index[1][0]
        self.detector.pixel_index_map = pixel_index_map.astype(np.int64)

    def get_pixel_corners(self):
        """
        Compute the pixel corners in the PyFAI reference frame
        """
        geo = self.detector.geo
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        x, y, z = geo.get_pixel_coords(oname=child.oname, oindex=0, do_tilt=True, cframe=0)
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

class PyFAIToPsana:
    """
    Class to convert PyFAI Detector instance to psana .data geometry files by using intermediate CrystFEL .geom files 

    Parameters
    ----------
    Parameters
    ----------
    in_file : str
        Path to the input .poni file containing geometry calibration parameters
    exp : str
        Experiment tag
    run_num : int
        Run number
    detname : str
        Detector name
    out_file : str
        Path to the output psana .data file
    """

    def __init__(self, in_file, exp, run_num, detname, out_file):
        converter = PsanaToPyFAI(exp=exp, run_num=run_num, detname=detname)
        self.detector = converter.detector
        ai = pyFAI.load(in_file)
        self.params = ai.param
        self.correct_geom()
        self.convert_pyfai_to_data(out_file=out_file)
    
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
        p1, p2, p3 = self.detector.calc_cartesian_positions()
        dist = self.params[0]
        poni1 = self.params[1]
        poni2 = self.params[2]
        p1 = (p1 - poni1 - (self.detector.pixel_size / 2)).ravel()
        p2 = (p2 - poni2 - (self.detector.pixel_size / 2)).ravel()
        if p3 is None:
            p3 = np.zeros_like(p1) + dist
        else:
            p3 = (p3+dist).ravel()
        coord_det = np.stack((p1, p2, p3), axis=0)
        coord_sample = np.dot(rotation_matrix(self.params), coord_det)
        x, y, z = coord_sample
        x, y, z = self.pyfai_to_psana(x, y, z, self.params)
        self.X = x
        self.Y = y
        self.Z = z

    def convert_pyfai_to_data(self, out_file):
        """
        Main function to convert PyFAI coordinates to psana .data geometry file

        Parameters
        ----------
        out_file : str
            Path to the output .data file
        """
        geo = self.detector.geo
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        topname = top.oname
        childname = child.oname
        npanels = self.detector.n_modules
        asics_shape = self.detector.asics_shape
        fs_size = self.detector.fs_size
        ss_size = self.detector.ss_size
        X = self.X.reshape(self.detector.raw_shape)
        Y = self.Y.reshape(self.detector.raw_shape)
        Z = self.Z.reshape(self.detector.raw_shape)
        recs = header_psana(detname=self.detector.detname)
        distance = self.params[0] * (1 / (np.cos(self.params[3] * np.cos(self.params[4]))))
        distance_um = round(distance * 1e6)
        for p in range(npanels):
            if npanels != 1:
                xp = X[p, :]
                yp = Y[p, :]
                zp = Z[p, :]
            else:
                xp = X
                yp = Y
                zp = Z
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
            vcent = (np.mean(xp), np.mean(yp), np.mean(zp)-distance_um)
            angle_deg_z = degrees(atan2(nfs[1], nfs[0]))
            angle_z, tilt_z = angle_and_tilt(angle_deg_z)
            tilt_x, tilt_y = tilt_xy(nfs, nss)
            angle_child_x = child.get_list_of_children()[p].rot_x
            angle_child_y = child.get_list_of_children()[p].rot_y
            angle_child_z = child.get_list_of_children()[p].rot_z
            angle_x, angle_y, tilt_x, tilt_y = rotate_z(angle_child_z, angle_child_x, angle_child_y, tilt_x, tilt_y)
            recs += '\n%12s  0 %12s %2d' %(childname, self.detector.segname, p)\
                +'  %8d %8d %8d %7.0f %6.0f %6.0f   %8.5f  %8.5f  %8.5f'%\
                (vcent[0], vcent[1], vcent[2], angle_z, angle_y, angle_x, tilt_z, tilt_y, tilt_x)
        recs += '\n%12s  0 %12s  0' %(topname, childname)\
            +'         0        0 %8d       0      0      0    0.00000   0.00000   0.00000' % (distance_um)
        f=open(out_file,'w')
        f.write(recs)
        f.close()

class PyFAIToCrystFEL:
    """
    Class to write CrystFEL .geom geometry files from PyFAI .poni files

    Parameters
    ----------
    in_file : str
        Path to the input .poni file containing geometry calibration parameters
    exp : str
        Experiment tag
    run_num : int
        Run number
    detname : str
        Detector name
    out_file : str
        Path to the output CrystFEL .geom file
    """

    def __init__(self, in_file, exp, run_num, detname, out_file):
        converter = PsanaToPyFAI(exp=exp, run_num=run_num, detname=detname)
        self.detector = converter.detector
        ai = pyFAI.load(in_file)
        self.params = ai.param
        self.correct_geom()
        self.convert_pyfai_to_geom(out_file=out_file)
    
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
        p1 = (p1 - poni1 - (self.detector.pixel_size / 2)).ravel()
        p2 = (p2 - poni2 - (self.detector.pixel_size / 2)).ravel()
        if p3 is None:
            p3 = np.zeros_like(p1) + dist
        else:
            p3 = (p3+dist).ravel()
        coord_det = np.stack((p1, p2, p3), axis=0)
        coord_sample = np.dot(rotation_matrix(params), coord_det)
        x, y, z = coord_sample
        x, y, z = self.pyfai_to_psana(x, y, z, params)
        self.X = x
        self.Y = y
        self.Z = z
    
    def convert_pyfai_to_geom(self, out_file):
        """
        From corrected X, Y, Z coordinates, write a CrystFEL .geom file

        Parameters
        ----------
        output_file : str
            Path to the output .geom file
        """
        x = self.X.reshape(self.detector.raw_shape)
        y = self.Y.reshape(self.detector.raw_shape)
        z = self.Z.reshape(self.detector.raw_shape)
        seg = self.detector.seg.algo
        nsegs = int(x.size/seg.size())
        arows, acols = seg.asic_rows_cols()
        srows, _ = seg.shape()
        pix_size = seg.pixel_scale_size()
        _, nasics_in_cols = seg.number_of_asics_in_rows_cols()
        nasicsf = nasics_in_cols
        txt = header_crystfel()
        for n in range(nsegs):
            txt = '\n'
            for a,(r0,c0) in enumerate(seg.asic0indices()):
                vfs = np.array((\
                    x[n,r0,c0+acols-1] - x[n,r0,c0],\
                    y[n,r0,c0+acols-1] - y[n,r0,c0],\
                    z[n,r0,c0+acols-1] - z[n,r0,c0]))
                vss = np.array((\
                    x[n,r0+arows-1,c0] - x[n,r0,c0],\
                    y[n,r0+arows-1,c0] - y[n,r0,c0],\
                    z[n,r0+arows-1,c0] - z[n,r0,c0]))
                nfs = vfs/np.linalg.norm(vfs)
                nss = vss/np.linalg.norm(vss)

                pref = '\np%da%d'%(n,a)

                txt +='%s/fs = %+.6fx %+.6fy %+.6fz' % (pref, nfs[0], nfs[1], nfs[2])\
                    + '%s/ss = %+.6fx %+.6fy %+.6fz' % (pref, nss[0], nss[1], nss[2])\
                    + '%s/res = %.3f' % (pref, 1e6/pix_size)\
                    + '%s/corner_x = %.6f' % (pref, x[n,r0,c0]/pix_size)\
                    + '%s/corner_y = %.6f' % (pref, y[n,r0,c0]/pix_size)\
                    + '%s/coffset = %.6f' % (pref, z[n,r0,c0]*1e-6)\
                    + '%s/min_fs = %d' % (pref, (a%nasicsf)*acols)\
                    + '%s/max_fs = %d' % (pref, (a%nasicsf+1)*acols-1)\
                    + '%s/min_ss = %d' % (pref, n*srows + (a//nasicsf)*arows)\
                    + '%s/max_ss = %d' % (pref, n*srows + (a//nasicsf+1)*arows - 1)\
                    + '%s/no_index = 0' % (pref)\
                    + '\n'
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
    exp : str
        Experiment tag
    run_num : int
        Run number
    detname : str
        Detector name
    out_file : str
        Path to the output psana .data file
    """
    def __init__(self, in_file, exp, run_num, detname, out_file):
        self.valid = False
        self.load_geom(in_file=in_file)
        ds = psana.DataSource(exp=exp, run=run_num)
        run = next(ds.runs())
        try:
            self.det = run.Detector(detname)
        except Exception as e:
            raise ValueError(f"Detector {detname} not found in run {run_num} of experiment {exp}. Error: {e}")
        self.convert_geom_to_data(detname=detname, out_file=out_file)

    def _parse_line_as_parameter(self, line):
        assert isinstance(line, str), 'line is not a str object'
        fields = line.split()
        nfields = len(fields)
        if fields[1] != '=':
            self.list_of_ignored_records.append(line)
            return
        keys = fields[0].split('/')
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
            resp = str_is_segment_and_asic(k0)
            if resp: k0=resp
            v = '' if nfields<3 else\
                sfields_to_xyz_vector(fields[2:]) if k1 in ('fs','ss') else\
                int(fields[2]) if k1 in ('max_ss', 'min_ss', 'max_fs', 'min_fs', 'no_index') else\
                int(fields[2]) if k1 in ('max_x', 'min_x', 'max_y', 'min_y') else\
                float(fields[2]) if k1 in ('res', 'corner_x', 'corner_y', 'adu_per_eV', 'coffset') else\
                float(fields[2]) if k1 in ('xfs', 'yfs', 'xss', 'yss') else\
                ' '.join(fields[2:])
            if k0 in self.dict_of_pars.keys():
                self.dict_of_pars[k0][k1] = v
            else:
                self.dict_of_pars[k0] = {k1:v,}
        else:
            self.list_of_ignored_records.append(line)
            return

    def load_geom(self, in_file):
        self.valid = False
        self.list_of_comments = []
        self.list_of_ignored_records = []
        self.dict_of_pars = {}
        f=open(in_file,'r')
        for linef in f:
            line = linef.strip('\n')
            if not line.strip(): continue
            if line[0] == ';':
                self.list_of_comments.append(line)
                continue
            self._parse_line_as_parameter(line)
        f.close()
        self.valid = True

    def geom_to_data(self, panelasics, out_file):
        geo = self.det.raw._det_geo()
        top = geo.get_top_geo()
        child = top.get_list_of_children()[0]
        topname = top.oname
        childname = child.oname
        seg = geo.get_seg_geo()
        sg = seg.algo
        segname = seg.oname
        X, Y, Z = sg.pixel_coord_array()
        PIX_SIZE_UM = sg.get_pix_size_um()
        M_TO_UM = 1e6
        xc0, yc0, _ = X[0,0], Y[0,0], Z[0,0]
        distance = self.dict_of_pars.get('coffset', 0)
        recs = header_psana(detname=self.det.raw._det_name())
        segz = np.array([self.dict_of_pars[k].get('coffset', 0) for k in panelasics.split(',')])
        meanroundz = round(segz.mean()*1e6)*1e-6
        distance += meanroundz
        for p, panel in enumerate(panelasics.split(',')):
            dicasic = self.dict_of_pars[panel]
            nfs = np.array(dicasic.get('fs', None), dtype=np.float64) 
            nss = np.array(dicasic.get('ss', None), dtype=np.float64) 
            vfs = nfs*abs(xc0)
            vss = nfs*abs(yc0)
            x0pix = dicasic.get('corner_x', 0)
            y0pix = dicasic.get('corner_y', 0)
            z0    = dicasic.get('coffset', 0)
            v00center = vfs + vss
            v00corner = np.array((x0pix*PIX_SIZE_UM, y0pix*PIX_SIZE_UM, (z0 - distance)*M_TO_UM))
            vcent = v00corner + v00center
            angle_deg_z = degrees(atan2(nfs[1], nfs[0]))
            angle_z, tilt_z = angle_and_tilt(angle_deg_z)
            tilt_x, tilt_y = tilt_xy(nfs, nss)
            angle_child_x = child.get_list_of_children()[p].rot_x
            angle_child_y = child.get_list_of_children()[p].rot_y
            angle_child_z = child.get_list_of_children()[p].rot_z
            angle_x, angle_y, tilt_x, tilt_y = rotate_z(angle_child_z, angle_child_x, angle_child_y, tilt_x, tilt_y)
            recs += '\n%12s  0 %12s %2d' %(childname, segname, p)\
                +'  %8d %8d %8d %7.0f %6.0f %6.0f   %8.5f  %8.5f  %8.5f'%\
                (vcent[0], vcent[1], vcent[2], angle_z, angle_y, angle_x, tilt_z, tilt_y, tilt_x)
        recs += '\n%12s  0 %12s  0' %(topname, childname)\
            +'         0        0 %8d       0      0      0    0.00000   0.00000   0.00000' % (distance*M_TO_UM)
        f=open(out_file,'w')
        f.write(recs)
        f.close()

    def convert_geom_to_data(self, detname, out_file):
        detname_lower = detname.lower()
        if "epix10kaquad" in detname_lower:
            panelasics = detname_to_pars.get("epix10kaquad", None)
        else:
            panelasics = detname_to_pars.get(detname_lower, None)
        self.geom_to_data(panelasics, out_file)

class CrystFELToPyFAI:
    """
    Class to convert CrystFEL .geom geometry files to PyFAI Detector instance by using intermediate psana .data files

    Parameters
    ----------
    in_file : str
        Path to the CrystFEL .geom file
    det_type : str
        Detector type
    psana_file : str
        Path to the psana .data file for retrieving segmentation information
    pixel_size : float
        Pixel size in meters
    shape : tuple
        Detector shape (n_modules, ss_size, fs_size)
    """
    def __init__(self, in_file, det_type, psana_file, pixel_size=None, shape=None):
        path = os.path.dirname(in_file)
        data_file = os.path.join(path, "temp.data")
        CrystFELToPsana(in_file=in_file, det_type=det_type, psana_file=psana_file, out_file=data_file)
        psana_to_pyfai = PsanaToPyFAI(in_file=data_file, det_type=det_type, pixel_size=pixel_size, shape=shape)
        self.detector = psana_to_pyfai.detector
        os.remove(data_file)