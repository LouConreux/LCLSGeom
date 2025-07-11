from pyFAI.detectors import Detector

class ePix10k2M(Detector):
    """
    PyFAI Detector instance for the ePix10k2M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.0001
        if shape is None:
            shape = (16, 352, 384)
        self.det_type = "epix10k2M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = 4
        self.asics_shape = (2, 2)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

class ePix10kaQuad(Detector):
    """
    PyFAI Detector instance for the ePix10kaQuad
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.0001
        if shape is None:
            shape = (4, 352, 384)
        self.det_type = "Epix10kaQuad"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = 4
        self.asics_shape = (2, 2)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

class Jungfrau05M(Detector):
    """
    PyFAI Detector instance for the Jungfrau05M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.0001
        if shape is None:
            shape = (512, 1024)
        self.det_type = "jungfrau05M"
        self.raw_shape = shape
        self.n_modules = 1
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

class Jungfrau1M(Detector):
    """
    PyFAI Detector instance for the Jungfrau1M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (2, 512, 1024)
        self.det_type = "jungfrau1M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), **kwargs)

class Jungfrau4M(Detector):
    """
    PyFAI Detector instance for the Jungfrau4M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (8, 512, 1024)
        self.det_type = "jungfrau1M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

class Jungfrau16M(Detector):
    """
    PyFAI Detector instance for the Jungfrau16M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (32, 512, 1024)
        self.det_type = "jungfrau16M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = 8
        self.asics_shape = (2, 4)
        self.ss_size = shape[1] // self.asics_shape[0]
        self.fs_size = shape[2] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

class Rayonix(Detector):
    """
    PyFAI Detector instance for the Rayonix
    By default, the Rayonix detector is defined unbinned. The user can specify the pixel size and detector shape to bin the detector if wanted.
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000176
        if shape is None:
            shape = (1920, 1920)
        self.det_type = "Rayonix"
        self.raw_shape = shape
        self.n_modules = 1
        self.n_asics = 1
        self.asics_shape = (1, 1)
        self.ss_size = shape[0] // self.asics_shape[0]
        self.fs_size = shape[1] // self.asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * self.asics_shape[0] * self.ss_size, self.asics_shape[1] * self.fs_size), orientation=0, **kwargs)

def get_detector(det_type, pixel_size=None, shape=None):
    """
    Instantiate a PyFAI Detector object based on the detector type

    Parameters
    ----------
    det_type : str
        Detector type
    """
    if det_type.lower() == "epix10k2m":
        Detector.registry["epix10k2M"] = ePix10k2M
        return ePix10k2M(pixel_size=pixel_size, shape=shape)
    elif "epix10kaquad" in det_type.lower():
        Detector.registry["epix10kaQuad"] = ePix10kaQuad
        return ePix10kaQuad(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau05m":
        Detector.registry["jungfrau05M"] = Jungfrau05M
        return Jungfrau05M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau1m":
        Detector.registry["jungfrau1M"] = Jungfrau1M
        return Jungfrau1M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau4m":
        Detector.registry["jungfrau4M"] = Jungfrau4M
        return Jungfrau4M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau16m":
        Detector.registry["jungfrau16M"] = Jungfrau16M
        return Jungfrau16M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "rayonix":
        Detector.registry["Rayonix"] = Rayonix
        return Rayonix(pixel_size=pixel_size, shape=shape)
    else:
        raise ValueError("Detector type not recognized")