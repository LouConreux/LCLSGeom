import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.swap_geom import PsanatoCrystFEL, CrystFELtoPyFAI, PyFAItoCrystFEL, CrystFELtoPsana
from pyFAI.geometry import Geometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.calibrant import CalibrantFactory, CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
import numpy as np
from scipy.signal import find_peaks

class mfxx49820_run8:
    def __init__(self):
        self.geomfile = "tests/data/geom/ePix10k2M_0-end.data"
        self.powder = "tests/data/powder/mfxx49820/calib_max_flat_r0008.npy"
        self.exp = "mfxx49820"
        self.run = 8
        self.det_type = "epix10k2M"
        self.dist = 0.282
        self.poni1 = -0.00079
        self.poni2 = -0.00067
        self.wavelength = 1.2910925615868107e-10
    
class cxil1019522_run5:
    def __init__(self):
        self.geomfile = "tests/data/geom/Jungfrau4M-0_end.data"
        self.powder = "tests/data/powder/cxil1019522/calib_max_flat_r0005.npy"
        self.exp = "cxil1019522"
        self.run = 5
        self.det_type = "jungfrau4M"
        self.dist = 0.147
        self.poni1 = 0.0
        self.poni2 = 0.0
        self.wavelength = 1.3992865102162315e-10

class mfxl1015222_run6:
    def __init__(self):
        self.geomfile = "tests/data/geom/Rayonix_0-end.data"
        self.powder = "tests/data/powder/mfxl1015222/calib_max_flat_r0006.npy"
        self.exp = "mfxl1015222"
        self.run = 6
        self.det_type = "Rayonix"
        self.dist = 0.104
        self.poni1 = 0.0
        self.poni2 = 0.0
        self.wavelength = 1.1047305314738256e-10

def generate_data_radial_integration(test):
    # Generate the pyFAI detector geometry object
    PsanatoCrystFEL(test.geomfile, test.geomfile.replace(".data", ".geom"), det_type=test.det_type)
    conv = CrystFELtoPyFAI(test.geomfile.replace(".data", ".geom"), psana_file=test.geomfile, det_type=test.det_type)
    det = conv.detector
    pixel_array = conv.pix_pos

    # Load powder data
    powder_img = np.load(test.powder)
    behenate = CALIBRANT_FACTORY('AgBh')
    behenate.wavelength = test.wavelength
    test.powder_img = powder_img

    # Optimize geometry both in translations and rotations
    geom_initial = Geometry(dist=test.dist, poni1=test.poni1, poni2=test.poni2, detector=det, wavelength=test.wavelength)
    sg = SingleGeometry("optimization", powder_img, calibrant=behenate, detector=det, geometry=geom_initial)
    sg.extract_cp(max_rings=5, pts_per_deg=1, Imin=np.max(powder_img)/100)
    score = sg.geometry_refinement.refine3(fix=["wavelength"])
    ai = sg.geometry_refinement.get_ai()
    q, I ,sigma = ai.integrate1d(powder_img, 1000, unit='q_A^-1' ,error_model="poisson")

    # Return peaks values
    peaks, _ = find_peaks(I, prominence=1, distance=10)
    return sg, pixel_array, q[peaks], I[peaks]

def test_pyFAItoCrystFEL(test, sg, pixel_array, q_peaks, I_peaks):
    # Convert the pyFAI geometry to CrystFEL
    conv = PyFAItoCrystFEL(sg=sg, pixel_array=pixel_array, psana_file=test.geomfile, output_file=test.geomfile.replace("0-end.data", "test.geom"))

    # Load the CrystFEL geometry
    conv = CrystFELtoPyFAI(geom_file=test.geomfile.replace("0-end.data", "test.geom"), det_type=test.det_type)
    det_test = conv.detector

    # Evaluate impact on radial integration
    dist = test.dist
    poni1 = 0
    poni2 = 0
    ai = AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2, detector=det_test, wavelength=test.wavelength)
    q, I, sigma = ai.integrate1d(test.powder_img, 1000, unit='q_A^-1', error_model="poisson")

    # Find peaks
    peaks, _ = find_peaks(I, prominence=1, distance=10)

    # Check if the peaks are the same
    print(f'Testing for {test.exp} run {test.run}')
    print('Test 1: After pyFAI to CrystFEL conversion')
    print(f'Are q values the same ?: {np.allclose(q[peaks], q_peaks)}')
    print(f'Are I values the same ?: {np.allclose(I[peaks], I_peaks)}')
    if np.allclose(q[peaks], q_peaks):
        print('Test 1: PASSED')

def test_CrystFELtoPsana(test, q_peaks, I_peaks):
    # Convert the CrystFEL geometry to psana
    conv = CrystFELtoPsana(geom_file=test.geomfile.replace("0-end.data", "test.geom"), det_type=test.det_type, psana_file=test.geomfile.replace("0-end.data", "test.data"))

    # Load the psana geometry
    conv = PsanatoCrystFEL(test.geomfile.replace("0-end.data", "test.data"), test.geomfile.replace("0-end.data", "test_test.geom"), det_type=test.det_type)
    conv = CrystFELtoPyFAI(test.geomfile.replace("0-end.data", "test_test.geom"), det_type=test.det_type)
    det_test = conv.detector

    # Evaluate impact on radial integration
    dist = test.dist
    poni1 = 0
    poni2 = 0
    ai = AzimuthalIntegrator(dist=dist, poni1=poni1, poni2=poni2, detector=det_test, wavelength=test.wavelength)
    q, I, sigma = ai.integrate1d(test.powder_img, 1000, unit='q_A^-1', error_model="poisson")

    # Find peaks
    peaks, _ = find_peaks(I, prominence=1, distance=10)

    # Check if the peaks are the same
    print(f'Testing for {test.exp} run {test.run}')
    print('Test 2: After CrystFEL to psana conversion')
    print(f'Are q values the same ?: {np.allclose(q[peaks], q_peaks)}')
    print(f'Are I values the same ?: {np.allclose(I[peaks], I_peaks)}')
    if np.allclose(q[peaks], q_peaks):
        print('Test 2: PASSED')

def main():
    test = mfxx49820_run8()
    sg, pixel_array, q_peaks, I_peaks = generate_data_radial_integration(test)
    test_pyFAItoCrystFEL(test, sg, pixel_array, q_peaks, I_peaks)
    test_CrystFELtoPsana(test, q_peaks, I_peaks)

    test = cxil1019522_run5()
    sg, pixel_array, q_peaks, I_peaks = generate_data_radial_integration(test)
    test_pyFAItoCrystFEL(test, sg, pixel_array, q_peaks, I_peaks)
    test_CrystFELtoPsana(test, q_peaks, I_peaks)

    test = mfxl1015222_run6()
    sg, pixel_array, q_peaks, I_peaks = generate_data_radial_integration(test)
    test_pyFAItoCrystFEL(test, sg, pixel_array, q_peaks, I_peaks)
    test_CrystFELtoPsana(test, q_peaks, I_peaks)

if __name__ == "__main__":
    main()

