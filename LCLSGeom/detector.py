from pyFAI.detectors import Detector

class ePix10k2M(Detector):
    """
    PyFAI Detector instance for the ePix10k2M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=4,
        asics_shape = (2, 2), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.0001
        if shape is None:
            shape = (16, 352, 384)
        self.det_type = "epix10k2M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[1] // asics_shape[0]
        self.fs_size = shape[2] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class ePix10kaQuad(Detector):
    """
    PyFAI Detector instance for the ePix10kaQuad
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=4,
        asics_shape = (2, 2), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.0001
        if shape is None:
            shape = (4, 352, 384)
        self.det_type = "Epix10kaQuad"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[1] // asics_shape[0]
        self.fs_size = shape[2] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class Jungfrau05M(Detector):
    """
    PyFAI Detector instance for the Jungfrau05M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (512, 1024)
        self.det_type = "jungfrau05M"
        self.raw_shape = shape
        self.n_modules = 1
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[0] // asics_shape[0]
        self.fs_size = shape[1] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class Jungfrau1M(Detector):
    """
    PyFAI Detector instance for the Jungfrau1M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (2, 512, 1024)
        self.det_type = "jungfrau1M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[1] // asics_shape[0]
        self.fs_size = shape[2] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class Jungfrau4M(Detector):
    """
    PyFAI Detector instance for the Jungfrau4M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (8, 512, 1024)
        self.det_type = "jungfrau4M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[1] // asics_shape[0]
        self.fs_size = shape[2] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class Jungfrau16M(Detector):
    """
    PyFAI Detector instance for the Jungfrau16M
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=8,
        asics_shape=(2, 4), # (rows, cols) = (ss, fs)
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000075
        if shape is None:
            shape = (32, 512, 1024)
        self.det_type = "jungfrau16M"
        self.raw_shape = shape
        self.n_modules = shape[0]
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[1] // asics_shape[0]
        self.fs_size = shape[2] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

class Rayonix(Detector):
    """
    PyFAI Detector instance for the Rayonix
    By default, the Rayonix detector is defined unbinned. The user can specify the pixel size and detector shape to bin the detector if wanted.
    """

    def __init__(
        self,
        pixel_size=None,
        shape=None,
        n_asics=1,
        asics_shape=(1, 1),
        **kwargs,
    ):
        if pixel_size is None:
            pixel_size = 0.000176
        if shape is None:
            shape = (1920, 1920)
        self.det_type = "Rayonix"
        self.raw_shape = shape
        self.n_modules = 1
        self.n_asics = n_asics
        self.asics_shape = asics_shape
        self.ss_size = shape[0] // asics_shape[0]
        self.fs_size = shape[1] // asics_shape[1]
        self.pixel_size = pixel_size
        super().__init__(pixel1=pixel_size, pixel2=pixel_size, max_shape=(self.n_modules * asics_shape[0] * self.ss_size, asics_shape[1] * self.fs_size), **kwargs)

def get_detector(det_type, pixel_size=None, shape=None):
    """
    Instantiate a PyFAI Detector object based on the detector type

    Parameters
    ----------
    det_type : str
        Detector type
    """
    if det_type.lower() == "epix10k2m":
        return ePix10k2M(pixel_size=pixel_size, shape=shape)
    elif "epix10kaquad" in det_type.lower():
        return ePix10kaQuad(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau05m":
        return Jungfrau05M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau1m":
        return Jungfrau1M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau4m":
        return Jungfrau4M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "jungfrau16m":
        return Jungfrau16M(pixel_size=pixel_size, shape=shape)
    elif det_type.lower() == "rayonix":
        return Rayonix(pixel_size=pixel_size, shape=shape)
    else:
        raise ValueError("Detector type not recognized")