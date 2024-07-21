# courtesy of E. Almamedov with modifications for our purposes

# import libraries
import numpy
import fabio

from base import utilities


""" Detector is the base class """
class Detector:
    MANUFACTURER = None
    MODEL = None
    VERSION = None

    uniform_pixel = True  # tells all pixels have the same size
    IS_FLAT = True  # this detector is flat
    IS_CONTIGUOUS = True  # No gaps: all pixels are adjacents, speeds-up calculation

    def __init__(self, maximum_shape, pixel1, pixel2):
        """
        :param maximum_shape: maximum size of the detector (:type 2-tuple of integers)
        :param pixel_1: size of the pixel in meter along the slow dimension (often Y) (:type: float)
        :param pixel_2: size of the pixel in meter along the fast dimension (often X) (:type: float)
        """
        self.maximum_shape = maximum_shape
        self.shape = self.maximum_shape
        self.pixel1 = pixel1
        self.pixel2 = pixel2
        self.distance = None
        self.wavelength = None

    def get_maximum_shape(self):
        return self.maximum_shape

    def get_pixel1(self):
        return self.pixel1

    def get_pixel2(self):
        return self.pixel2

    def get_type(self):
        return self.MANUFACTURER + " " + self.MODEL + self.VERSION


""" Dectris (subclass)( inheritance from Detector (base class)"""
class Dectris(Detector):

    MANUFACTURER = "Dectris"

    def __init__(self, maximum_shape, pixel1, pixel2):
        Detector.__init__(self, maximum_shape=maximum_shape, pixel1=pixel1, pixel2=pixel2)

""" Eiger (subclass) inheritance from Dectris (base class)"""
class Eiger(Dectris):
    """
    Eiger detector: generic description containing mask algorithm

    Note: 512k modules (514*1030) are made of 2x4 submodules of 256*256 pixels.
    Two missing pixels are interpolated at each sub-module boundary which explains
    the +2 and the +6 pixels.
    """

    MODEL = "Eiger"
    MODULE_SIZE = (514, 1030)
    MODULE_GAP = (37, 10)

    def __init__(self, maximum_shape, pixel1=75e-6, pixel2=75e-6):
        Dectris.__init__(self, maximum_shape=maximum_shape, pixel1=pixel1, pixel2=pixel2)
        self.maximum_shape = maximum_shape
        self.offset1 = self.offset2 = None

    def get_maximum_shape(self):
        return self.maximum_shape

    def calculate_mask(self):
        """
        Returns a generic mask for Pilatus detectors...
        """
        mask = numpy.zeros(self.maximum_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.MODULE_SIZE[0], self.maximum_shape[0], self.MODULE_SIZE[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.MODULE_SIZE[1], self.maximum_shape[1], self.MODULE_SIZE[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

    def get_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        :param d1: the Y pixel positions (slow dimension) (:type ndarray (1D or 2D))
        :param d2: the X pixel positions (fast dimension) (:type ndarray (1D or 2D))

        :return: p1, p2 position in meter of the center of each pixels. (:rtype: 2-tuple of numpy.ndarray)

        d1 and d2 must have the same shape, returned array will have the same shape.
        """
        if self.maximum_shape:
            if (d1 is None) or (d2 is None):
                d1 = utilities.expand2d(numpy.arange(self.maximum_shape[0]).astype(numpy.float32), self.maximum_shape[1], False)
                d2 = utilities.expand2d(numpy.arange(self.maximum_shape[1]).astype(numpy.float32), self.maximum_shape[0], True)

        if self.offset1 is None or self.offset2 is None:
            delta1 = delta2 = 0.
        else:
            if d2.ndim == 1:
                d1n = d1.astype(numpy.int32)
                d2n = d2.astype(numpy.int32)
                delta1 = self.offset1[d1n, d2n] / 100.0  # Offsets are in percent of pixel
                delta2 = self.offset2[d1n, d2n] / 100.0
            else:
                if d1.shape == self.offset1.shape:
                    delta1 = self.offset1 / 100.0  # Offsets are in percent of pixel
                    delta2 = self.offset2 / 100.0
                elif d1.shape[0] > self.offset1.shape[0]:  # probably working with corners
                    s0, s1 = self.offset1.shape
                    delta1 = numpy.zeros(d1.shape, dtype=numpy.int32)  # this is the natural type for pilatus CBF
                    delta2 = numpy.zeros(d2.shape, dtype=numpy.int32)
                    delta1[:s0, :s1] = self.offset1
                    delta2[:s0, :s1] = self.offset2
                    mask = numpy.where(delta1[-s0:, :s1] == 0)
                    delta1[-s0:, :s1][mask] = self.offset1[mask]
                    delta2[-s0:, :s1][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[-s0:, -s1:] == 0)
                    delta1[-s0:, -s1:][mask] = self.offset1[mask]
                    delta2[-s0:, -s1:][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[:s0, -s1:] == 0)
                    delta1[:s0, -s1:][mask] = self.offset1[mask]
                    delta2[:s0, -s1:][mask] = self.offset2[mask]
                    delta1 = delta1 / 100.0  # Offsets are in percent of pixel
                    delta2 = delta2 / 100.0  # former arrays were integers
                else:
                    print("Surprising situation !!! please investigate: offset has shape %s and input array have %s", self.offset1.shape, d1.shape)
                    delta1 = delta2 = 0.
        if center:
            # Eiger detectors images are re-built to be contiguous
            delta1 += 0.5
            delta2 += 0.5
        # For Eiger,
        p1 = (self.pixel1 * (delta1 + d1))
        p2 = (self.pixel2 * (delta2 + d2))
        return p1, p2, None

""" EigerX (subclass) inheritance from Eiger (base class)"""
class Eiger500k(Eiger):
    MAXIMUM_SHAPE = (512, 1030)
    VERSION = "500k"

    def __init__(self):
        Eiger.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

class Eiger1M(Eiger):
    MAXIMUM_SHAPE = (1065, 1030)
    VERSION = "500k"

    def __init__(self):
        Eiger.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

class Eiger4M(Eiger):
    MAXIMUM_SHAPE = (2167, 2070)
    VERSION = "4m"

    def __init__(self, mask="Eiger_4M.tif"):
        Eiger.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)
        self.maskfile = mask

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()

class Eiger9M(Eiger):
    MAXIMUM_SHAPE = (3269, 3110)
    VERSION = "9M"

    def __init__(self, mask="Eiger_X_9M.tif"):
        Eiger.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)
        self.maskfile = mask

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()


class Eiger16M(Eiger):
    MAXIMUM_SHAPE = (4371, 4150)
    VERSION = "16M"

    def __init__(self):
        Eiger.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)


class Pilatus(Dectris):
    """
     Pilatus detector: generic description containing mask algorithm

     Sub-classed by Pilatus1M, Pilatus2M and Pilatus6M
     """
    MODEL = "Pilatus"
    MODULE_SIZE = (195, 487)
    MODULE_GAP = (17, 7)

    def __init__(self, maximum_shape, pixel1=172e-6, pixel2=172e-6):
        Dectris.__init__(self, maximum_shape=maximum_shape, pixel1=pixel1, pixel2=pixel2)
        self.maximum_shape = maximum_shape
        self.offset1 = self.offset2 = None

    def calculate_mask(self):
        """
        Returns a calculated generic mask for Pilatus detectors...
        """
        mask = numpy.zeros(self.maximum_shape, dtype=numpy.int8)
        # workinng in dim0 = Y
        for i in range(self.MODULE_SIZE[0], self.maximum_shape[0], self.MODULE_SIZE[0] + self.MODULE_GAP[0]):
            mask[i: i + self.MODULE_GAP[0], :] = 1
        # workinng in dim1 = X
        for i in range(self.MODULE_SIZE[1], self.maximum_shape[1], self.MODULE_SIZE[1] + self.MODULE_GAP[1]):
            mask[:, i: i + self.MODULE_GAP[1]] = 1
        return mask

    def get_cartesian_positions(self, d1=None, d2=None, center=True, use_cython=True):
        """
        Calculate the position of each pixel center in cartesian coordinate and in meter of a couple of coordinates.
        The half pixel offset is taken into account here !!!

        :param d1: the Y pixel positions (slow dimension) (:type: ndarray (1D or 2D))
        :param d2: the X pixel positions (fast dimension) (:type: ndarray (1D or 2D))

        :return: position in meter of the center of each pixels. (:rtype: ndarray)

        d1 and d2 must have the same shape, returned array will have the same shape.
        """
        if self.maximum_shape and ((d1 is None) or (d2 is None)):
            d1 = utilities.expand2d(numpy.arange(self.maximum_shape[0]).astype(numpy.float32), self.maximum_shape[1], False)
            d2 = utilities.expand2d(numpy.arange(self.maximum_shape[1]).astype(numpy.float32), self.maximum_shape[0], True)

        if (self.offset1 is None) or (self.offset2 is None):
            delta1 = delta2 = 0.
        else:
            if d2.ndim == 1:
                d1n = d1.astype(numpy.int32)
                d2n = d2.astype(numpy.int32)
                delta1 = -self.offset1[d1n, d2n] / 100.0  # Offsets are in percent of pixel and negative
                delta2 = -self.offset2[d1n, d2n] / 100.0
            else:
                if d1.shape == self.offset1.shape:
                    delta1 = -self.offset1 / 100.0  # Offsets are in percent of pixel and negative
                    delta2 = -self.offset2 / 100.0
                elif d1.shape[0] > self.offset1.shape[0]:  # probably working with corners
                    s0, s1 = self.offset1.shape
                    delta1 = numpy.zeros(d1.shape, dtype=numpy.int32)  # this is the natural type for pilatus CBF
                    delta2 = numpy.zeros(d2.shape, dtype=numpy.int32)
                    delta1[:s0, :s1] = self.offset1
                    delta2[:s0, :s1] = self.offset2
                    mask = numpy.where(delta1[-s0:, :s1] == 0)
                    delta1[-s0:, :s1][mask] = self.offset1[mask]
                    delta2[-s0:, :s1][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[-s0:, -s1:] == 0)
                    delta1[-s0:, -s1:][mask] = self.offset1[mask]
                    delta2[-s0:, -s1:][mask] = self.offset2[mask]
                    mask = numpy.where(delta1[:s0, -s1:] == 0)
                    delta1[:s0, -s1:][mask] = self.offset1[mask]
                    delta2[:s0, -s1:][mask] = self.offset2[mask]
                    delta1 = -delta1 / 100.0  # Offsets are in percent of pixel and negative
                    delta2 = -delta2 / 100.0  # former arrays were integers
                else:
                    print("Surprizing situation !!! please investigate: offset has shape %s and input array have %s", self.offset1.shape, d1.shape)
                    delta1 = delta2 = 0.
        # For Pilatus,
        if center:
            # Account for the pixel center: pilatus detector are contiguous
            delta1 += 0.5
            delta2 += 0.5
        p1 = (self.pixel1 * (delta1 + d1))
        p2 = (self.pixel2 * (delta2 + d2))
        return p1, p2, None

class Pilatus100k(Pilatus):
    MAXIMUM_SHAPE = (195, 487)
    VERSION = "100k"

    def __init__(self):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

class Pilatus200k(Pilatus):
    MAXIMUM_SHAPE = (407, 487)
    VERSION = "200k"

    def __init__(self):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

class Pilatus300k(Pilatus):
    MAXIMUM_SHAPE = (619, 487)
    VERSION = "300k"

    def __init__(self, mask="Pilatus_300K.tif"):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)
        self.maskfile = mask

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()

class Pilatus300kw(Pilatus):
    MAXIMUM_SHAPE = (195, 1475)
    VERSION = "300kw"

    def __init__(self):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

class Pilatus1M(Pilatus):
    MAXIMUM_SHAPE = (1043, 981)
    VERSION = "1M"

    def __init__(self, mask="Pilatus_1M.tif"):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)
        self.maskfile = mask

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()

class Pilatus2M(Pilatus):
    MAXIMUM_SHAPE = (1679, 1475)
    VERSION = "2M"

    def __init__(self, mask="Pilatus_2M.tif"):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()

class Pilatus6M(Pilatus):
    MAXIMUM_SHAPE = (2527, 2463)
    VERSION = "6M"

    def __init__(self, mask="Pilatus_6M.tif"):
        Pilatus.__init__(self, maximum_shape=self.MAXIMUM_SHAPE)

    def load_mask(self):
        if mask:
            return fabio.open(self.maskfile).data
        else:
            return self.load_mask()
