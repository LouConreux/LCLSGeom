from pyFAI.detectors import Detector

class ePix10k2M(Detector):
    """
    PyFAI Detector instance for the ePix10k2M
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.0001
        self.detname = "epix10k2M"
        self.raw_shape = (16, 352, 384)
        self.n_modules = self.raw_shape[0]
        self.n_asics = 4
        self.asics_shape = (2, 2)
        self.ss_size = self.raw_shape[1] // self.asics_shape[0]
        self.fs_size = self.raw_shape[2] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

class ePix10kaQuad(Detector):
    """
    PyFAI Detector instance for the ePix10kaQuad
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.0001
        self.raw_shape = (4, 352, 384)
        self.detname = "Epix10kaQuad"
        self.n_modules = self.raw_shape[0]
        self.n_asics = 4
        self.asics_shape = (2, 2)
        self.ss_size = self.raw_shape[1] // self.asics_shape[0]
        self.fs_size = self.raw_shape[2] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

class Jungfrau05M(Detector):
    """
    PyFAI Detector instance for the Jungfrau05M
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.000075
        self.raw_shape = (512, 1024)
        self.detname = "jungfrau05M"
        self.n_modules = 1
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = self.raw_shape[0] // self.asics_shape[0]
        self.fs_size = self.raw_shape[1] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

class Jungfrau1M(Detector):
    """
    PyFAI Detector instance for the Jungfrau1M
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.000075
        self.raw_shape = (2, 512, 1024)
        self.detname = "jungfrau1M"
        self.n_modules = self.raw_shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = self.raw_shape[1] // self.asics_shape[0]
        self.fs_size = self.raw_shape[2] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

class Jungfrau4M(Detector):
    """
    PyFAI Detector instance for the Jungfrau4M
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.000075
        self.raw_shape = (8, 512, 1024)
        self.detname = "jungfrau4M"
        self.n_modules = self.raw_shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = self.raw_shape[1] // self.asics_shape[0]
        self.fs_size = self.raw_shape[2] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

class Jungfrau16M(Detector):
    """
    PyFAI Detector instance for the Jungfrau16M
    """

    def __init__(
        self,
    ):
        self.pixel_size = 0.000075
        self.raw_shape = (32, 512, 1024)
        self.detname = "jungfrau16M"
        self.n_modules = self.raw_shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = self.raw_shape[1] // self.asics_shape[0]
        self.fs_size = self.raw_shape[2] // self.asics_shape[1]
        super().__init__(pixel1=self.pixel_size, pixel2=self.pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size))

def get_detector(shape):
    """
    Instantiate a PyFAI Detector object based on stacked shape.

    Parameters
    ----------
    shape : tuple
        Stacked shape of the detector
    """
    if shape == (16, 352, 384):
        Detector.registry["epix10k2m"] = ePix10k2M
        return ePix10k2M()
    elif shape == (4, 352, 384):
        Detector.registry["epix10kaquad"] = ePix10kaQuad
        return ePix10kaQuad()
    elif shape == (1, 512, 1024):
        Detector.registry["jungfrau05m"] = Jungfrau05M
        return Jungfrau05M()
    elif shape == (2, 512, 1024):
        Detector.registry["jungfrau1m"] = Jungfrau1M
        return Jungfrau1M()
    elif shape == (8, 512, 1024):
        Detector.registry["jungfrau4m"] = Jungfrau4M
        return Jungfrau4M()
    elif shape == (32, 512, 1024):
        Detector.registry["jungfrau16m"] = Jungfrau16M
        return Jungfrau16M()
    else:
        raise ValueError("Detector type not recognized")