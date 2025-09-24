# -------------------------------------
#
# PostProduction/ImageIO.py
#
# Reading raw images (Nikon NEF), converting to other formats.
#
# Last update: January 2025
# Last author: Jessica Baron
#
# -------------------------------------

import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
# os.environ["COLOUR_SCIENCE__COLOUR__IMPORT_VAAB_COLOUR"] = "1"

import glob
import time
from pathlib import Path

import colour
import cv2 as cv
import numpy as np
import skimage.transform as skt
from PIL import Image

# import imageio as iio
# iio.plugins.freeimage.download()
# From copying over lib/python/site-packages/OpenImageIO from brew install openimageio
# BUT WITH WRONG ARCHITECTURE
#import OpenImageIO as oiio

from .Math import printStats

PIXEL_MAX = 2000

try:
    import OpenEXR
    def exr_read(path: Path, verbose: bool=False) -> np.ndarray:
        """Read EXR using OpenEXR.

        <https://openexr.com/en/latest/python.html>
        """
        with OpenEXR.File(str(path)) as exrFile:
            channelDict = exrFile.channels()
            
            if verbose:
                print('exr_read(): Header fields')
                part = exrFile.parts[0]
                for name, value in part.header.items():
                    print(name, value)
                # For example...
                # channels [Channel("B", xSampling=1, ySampling=1), Channel("G", xSampling=1, ySampling=1), Channel("R", xSampling=1, ySampling=1)]
                # compression Compression.PIZ_COMPRESSION
                # dataWindow (array([0, 0], dtype=int32), array([1079, 1090], dtype=int32))
                # displayWindow (array([0, 0], dtype=int32), array([1079, 1090], dtype=int32))
                # lineOrder LineOrder.INCREASING_Y
                # pixelAspectRatio 1.0
                # screenWindowCenter [ 0.  0.]
                # screenWindowWidth 1.0

            if len(channelDict) > 1:
                print("EXR mild warning; multiple channels to choose: ", channelDict.keys())
            
            # Choose whichever channel comes first
            channel = next(iter(channelDict.values()))

            # 'RGB' is considered *ONE* channel, so this will return an RGB image.
            # Also convert from float16 HalfEXR images to float32 to do computations at higher precision.
            return channel.pixels.astype(np.float32)

    def exr_write(path: Path, img: np.ndarray):
        """Write EXR using OpenEXR.
        It would be possible to have multiple channels if we wanted!
        <https://openexr.com/en/latest/python.html>
        """
        #print('exr_write(): ', path)

        # IMPORTANT! Always change to float16 (HalfEXR) precision because cameras capture no more than 16-bit.
        img = img.astype(np.float16)
        
        # The memory layout must be affecting EXR.
        # Applying np.ascontiguousarray() to force a dense order of values in memory.
        # A np.array is not guaranteed to be dense/contigous in memory but could consist of pointers to multiple areas.
        img = np.ascontiguousarray(img)

        header = {
                    #"compression" : OpenEXR.ZIP_COMPRESSION,
                    "compression" : OpenEXR.PIZ_COMPRESSION,
                    "type" : OpenEXR.scanlineimage
                 }
                
        # An RGB image becomes *ONE* 'RGB' EXR channel.
        channels = { "RGB" : img }

        with OpenEXR.File(header, channels) as file:
            file.write(str(path))
            
        return

except:
    raise ImportError('Need OpenEXR')
# except ImportError:
#     print('Utilities/ImageIO.py: WARNING: Using colour package for EXR writing instead of OpenEXR. (Unreliable because'
#     'this requires OpenImageIO or imageio installed without conflicting with OpenCV.)')
#     def exr_read(path: Path) -> np.ndarray:
#         return colour.read_image(path, 'float32')
#
#     def exr_write(path: Path, img: np.ndarray):
#         colour.write_image(img, str(path), method='OpenImageIO')



# RawPy gamma params: 'gamma' and 'slope'
# 20.0, 20.0    for lighter skin
# 20.0, 30.0    for darker skin

# TODO Confirm accuracy, rethink organization, update datatypes...
# input data type, output bit depth (Some pairing here relies on how these become params for rawpy.postprocess().)
IMAGE_FORMAT_TO_BIT_DEPTH = {
    'nef' : ('float32', 16),  # 12-bit NEF to 16-bit out of rawpy to float32 for further processing.
    'exr' : ('float32', 32),
    'jpg' : ('uint8', 8),
    'png' : ('uint8', 8),
}

# imageio.imsave(path.split("/")[-1].split(".")[0] + ".exr", img)

# rotate image 90deg
def rotate90(img):
    img = np.moveaxis(img, [0, 1, 2], [1, 0, 2])  # swap rows/columns
    img = np.flip(img, axis=1)   # negate some mirroring that happens from the swap
    return img


# Read image EXIF metadata as a dictionary.
def readImageEXIF(imagePath):
    from PIL import ExifTags, Image
    img = Image.open(imagePath)
    #exif = img._getexif()
    # https://stackoverflow.com/questions/4764932/in-python-how-do-i-read-the-exif-data-for-an-image
    exifDict = {
        ExifTags.TAGS[k]: v
        for k, v in img._getexif().items()
        if k in ExifTags.TAGS
    }
    return exifDict
   
    
def readRaw(path, rawpyGammaParams=None, rawpyNoAutobright=True):
    import rawpy
    dtype, outputBits = IMAGE_FORMAT_TO_BIT_DEPTH['nef']
    
    if rawpyGammaParams is None:
        with rawpy.imread(path) as raw:
            # Returns uint16 array for output bit depth 16.
            # Look at all parameter options here: https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
            # TODO Should we manually give white balance? What does rawpy.postprocess do by default?
            img = raw.postprocess(gamma=(1, 1),
                                  no_auto_bright=rawpyNoAutobright,
                                  output_bps=outputBits)
        # Make sure to keep array dtype matching the given outputBits.
        # Python 'float' is equivalent to np.float64.
        img = img.astype(dtype) / 65535   # Scale to 0..1 by 2^16.  (The division makes a float32 no matter the initial array type.)
        #img = img.astype(np.float32)      # Just keep it as a float32 more easily used throughout processing...
    else:
        with rawpy.imread(path) as raw:
            # Output bits are 8 by default.
            img = raw.postprocess(gamma=rawpyGammaParams,
                                  no_auto_bright=rawpyNoAutobright,
                                  output_bps=outputBits)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    return img



# From https://github.com/rgl-epfl/brdf-loader/blob/master/python/visualize.py
def RGL_to_srgb(v):
    return np.where(v < 0.0031308, v * 12.92,
                    1.055 * (v ** (1 / 2.4)) - 0.055)

def RGL_tonemap(img):
    tonemapped = img * (1.0 / np.maximum(1e-10, img.max()))
    tonemapped = np.nan_to_num(tonemapped, copy=False, nan=0.5, posinf=1, neginf=0)
    tonemapped = RGL_to_srgb(np.clip(tonemapped, 0, 1))
    return tonemapped

def RGL_tonemap_uint8(img):
    return (RGL_tonemap(img).clip(0, 1) * 255).astype(np.uint8)


# TODO Update this code for getting more flexible reading and changing data types if desired.
def readImage(path, inExt=None, outDType=None, bgr2rgb=False,
              rawpyGammaParams=None, rawpyNoAutobright=True, tonemap=False) -> np.ndarray:
    
    if type(path) is not str:
        path = str(path)
    
    #print('readImage() for path: ', path)
    if inExt is None:
        inExt = str(Path(path).suffix).replace('.', '')

    if inExt == 'nef':
        img = readRaw(path, rawpyGammaParams, rawpyNoAutobright)
        
    # EXRs from colour_hdri and originally poorly exposed images produces a lot of NaNs.
    elif inExt == 'exr':
        img = exr_read(path)
        #img = exr_read(path, verbose=True)
        img = np.nan_to_num(img)   # Get rid of any occasional NaNs from HDRs. KEEP THIS.
        
    else:
        img = cv.imread(path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)   # OpenCV swaps rgb,bgr by default.

    if img is None:
        print('WARNING. Cannot read image. Returning None.')
        return None

    if tonemap:
        img = RGL_tonemap(img)
    
    if bgr2rgb:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    if outDType is not None:
        img = colour.io.convert_bit_depth(img, outDType)

    #print(f'\tReturning image data of dtype {img.dtype} and shape {img.shape}.')
    return img


def infoString(img):
    return f'shape={img.shape}, dtype={img.dtype},' \
        f'\t min={img.min()}, max={img.max()}, avg={np.mean(img)}, std={np.std(img)}'

# Old idea. Use colour.io.convert_bit_depth()?
def float32_to_uint8(img, shiftByAverage=False):
    mn, mx = img.min(), img.max()
   
    if shiftByAverage:
        avg = np.average(img)
        med = np.median(img)
        #buf = (mx - avg) * 0.5
        upper = med
        lower = mn
        print('min, max', mn, mx)
        print('avg, med', avg, med)
        print('lower, upper', lower, upper)
        img = (255 * ((img - lower) / (upper - lower))).astype(np.uint8)

    else:
        img = (255 * ((img - mn) / (mx - mn))).astype(np.uint8)
        
    return img

def downscale(img, factor):
    w, h = img.shape[1], img.shape[0]
    img = cv.resize(img, (w // factor, h // factor))
    return img

# Something quick
def fakeTonemap(img, gamma=2.2):
    dtype = img.dtype
    img = np.power(img, 1/gamma).astype(dtype)
    return img

# Multiple functions available: https://colour-hdri.readthedocs.io/en/develop/colour_hdri.tonemapping.html
def tonemapHDR(img, gamma=True, exp=False, params=(2.2, 0.0)):
    import colour_hdri
    if gamma:
        img = colour_hdri.tonemapping_operator_gamma(img, gamma=params[0], EV=params[1])
    if exp:
        img = colour_hdri.tonemapping_operator_exponentiation_mapping(img, p=params[0], q=params[1])
    #img = colour_hdri.tonemapping_operator_normalisation(img)
    #img = colour_hdri.tonemapping_operator_logarithmic_mapping(img)
    return img

# percent = 0.5 for half resolution
def resizeByPercentage(image, percent):
    hOld, wOld = image.shape[0:2]
    hNew = int(percent * hOld)
    wNew = int(percent * wOld)
    newShape = (hNew, wNew, 3)
    image = skt.resize_local_mean(image, newShape)
    image = image.reshape(newShape)  # Force again
    return image

def cropAndResize(image, cropBounds, resizeRes=None, forceSquare=False):
    x0, x1, y0, y1 = cropBounds
    image = image[y0:y1, x0:x1]
    if resizeRes is not None:
        hOld, wOld = image.shape[0:2]
        aspect = hOld / wOld
        if forceSquare:
            wNew = hNew = resizeRes
        else:
            hNew = resizeRes
            wNew = np.around(resizeRes * aspect).astype(np.int32)
        newShape = (hNew, wNew, 3)
        # resize_local_mean() is more data preserving?
        #image = skt.resize(image, newShape, anti_aliasing=False)
        
        # TODO Writing the resized version is not working on Countenance?? dtype still float32 and values still same...
        image = skt.resize_local_mean(image, newShape)
        image = image.reshape(newShape)   # Force shape again

    return image


def resize(image, newWidth, newHeight, antiAlias=False):
    newShape = (newHeight, newWidth, 3)
    
    if antiAlias:
        image = skt.resize(image, newShape, anti_aliasing=True)
        
    else:   # Most data-preserving method?
        image = skt.resize_local_mean(image, newShape)
    
    return image


def normalizeImage(img):
    mn = img.min()
    mx = img.max()
    return (img - mn) / (mx - mn)

def getAverageGrayscaleValue(img):
    r = img[:, 0]
    g = img[:, 1]
    b = img[:, 2]
    allGray = (r + g + b) / 3
    return np.average(allGray)


# row by row
def generatePixelIndices(imageWidth, imageHeight, flatten=True):
    rows, cols = np.meshgrid(np.arange(0, imageHeight),
                             np.arange(0, imageWidth))
    locs = np.array([rows.ravel(), cols.ravel()]).T
    if flatten:
        return locs
    else:
        return locs.reshape((imageHeight, imageWidth, 2))


def findAveragePixelInRegion(img, bounds01):
    h, w, c = img.shape
    x0 = int(w * bounds01[0])
    x1 = int(w * bounds01[1])
    y0 = int(h * bounds01[2])
    y1 = int(h * bounds01[3])
    slice = img[y0:y1, x0:x1]
    val = np.average(slice, axis=(0, 1))
    return val


def cropToCircle(img, regionRadius01, center01=[0.5, 0.5], centerInd1D=None, returnMaskedImage=False):
    h, w = np.array(img.shape[:2])
    radius = (h * regionRadius01).astype(np.int32)
    
    # Generate all combinations of (column, row) pixel indices as an (w*h, 2) array.
    pixelInds = generatePixelIndices(h, w)

    if centerInd1D is None:
        center01 = np.array(center01)
        x = int(center01[0] * w)
        y = int(center01[1] * h)
        centerInd = np.array([x, y])
    else:
        x = centerInd1D % w
        y = centerInd1D // w
        centerInd = np.array([x, y])

    diff = pixelInds - centerInd
    dist = np.sqrt(diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1])
    inCircle = (dist < radius).reshape((h, w))
 
    pixelValuesInCircle = img[inCircle]
    pixelInds = pixelInds.reshape((h, w, 2))   # Reshape like the image.
    pixelIndsInCircle = pixelInds[inCircle]

    if returnMaskedImage:
        #v = np.median(img)
        v = 1.0
        gray = np.ones(img.shape) * np.array([v, v, v])
        gray[inCircle] = img[inCircle]
        return pixelValuesInCircle, pixelIndsInCircle, gray
    else:
        return pixelValuesInCircle, pixelIndsInCircle


# Similar setup as cropToCircle().
def cropToRect(img, regionRadius01, center01=[0.5, 0.5], centerInd1D=None, returnMaskedImage=False, doSquare=True):
    H, W = np.array(img.shape[:2])
    hHalf = (H * regionRadius01).astype(np.int32)
    if doSquare:
        wHalf = hHalf
    else:
        wHalf = (W * regionRadius01).astype(np.int32)

    # Generate all combinations of (column, row) pixel indices as an (w*h, 2) array.
    pixelInds = generatePixelIndices(H, W)
    
    if centerInd1D is None:
        center01 = np.array(center01)
        x = int(center01[0] * W)
        y = int(center01[1] * H)
        centerInd = np.array([x, y])
    else:
        x = centerInd1D % W
        y = centerInd1D // W
        centerInd = np.array([x, y])
    
    #distX = np.abs(centerInd[0] - pixelInds[:, 0])
    #distY = np.abs(centerInd[1] - pixelInds[:, 1])
    #inW = (distX <= wHalf).reshape((H, W))
    #inH = (distY <= hHalf).reshape((H, W))
    #inRect = np.logical_and(inW, inH).reshape((H, W))

    x0 = centerInd[0] - wHalf
    y0 = centerInd[1] - hHalf
    x1 = centerInd[0] + wHalf
    y1 = centerInd[1] + hHalf
    inW = np.logical_and(x0 <= pixelInds[:, 0], pixelInds[:, 0] < x1).reshape((H, W))
    inH = np.logical_and(y0 <= pixelInds[:, 1], pixelInds[:, 1] < y1).reshape((H, W))
    inRect = np.logical_and(inW, inH).reshape((H, W))
    
    # Since rectangular can return as a new image of shape (hHalf*2, wHalf*2, 3) but might not be exact.
    w = x1 - x0
    h = y1 - y0
    newImg = img[inRect].reshape(h, w, 3)
    
    pixelValuesInRect = img[inRect]
    pixelInds = pixelInds.reshape((H, W, 2))   # Reshape like the image.
    pixelIndsInRect = pixelInds[inRect]
   
    if returnMaskedImage:
        v = 1.0
        gray = np.ones(img.shape) * np.array([v, v, v])
        # Since rectangular can return as a new image of shape (hHalf*2, wHalf*2, 3) but might not be exact.
        gray[inRect] = img[inRect]
        return pixelValuesInRect, pixelIndsInRect, gray, newImg
    else:
        return pixelValuesInRect, pixelIndsInRect, newImg

    return


def convertImageFormat(img, outputType):
    #print(f'\tInput image data: {infoString(img)}')
    if outputType == 'uint8':
        img = float32_to_uint8(img)   # TODO Needs help.
    else:
        img = colour.io.convert_bit_depth(img, outputType)
    #print(f'\tOutput image data: {infoString(img)}')
    return img


def writeImage(img, savePath, extOut=None, verbose:bool = True):
    if extOut is not None:
        outputType, outputBits = IMAGE_FORMAT_TO_BIT_DEPTH[extOut]
        img = convertImageFormat(img, outputType)
        #print(f"writeImage() with output extension '{extOut}' of bit type/depth {outputType}, {outputBits} bits.")
    #print(f'\tSaving to: {savePath}')
    else:
        extOut = Path(savePath).suffix[1:].lower()

    # OpenCV seems more reliable for JPGs?
    if extOut.lower() in {"jpg", "png", "webp"}:
        if verbose:
            printStats(img, 'img')
        #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        #cv.imwrite(savePath, img)
        if img.dtype != np.uint8:
            img = (img / img.max()) * 255
            
        imgPIL = Image.fromarray(img.astype(np.uint8), 'RGB')
        imgPIL.save(savePath)
        return img
        
    elif extOut == 'exr':
        exr_write(savePath, img)
        
    else:
        # Ideally want for all formats but unpredictable backend installations...
        # Colour Science relies on different image backends per OS.
        colour.write_image(img, str(savePath), method='OpenImageIO')
    #colour.write_image(img, savePath, bit_depth='float32', method='OpenImageIO')

        return img


# Work-in-progress for more image formats.
def convertImageExtensions(path, extIn, extOut, rawpyGammaParams=None, rawpyNoAutobright=True,
                           downsample=False, tonemap=False, bgr2rgb=False):
    print('convertImageFormat(). WARNING: extOut PARAM CURRENTLY NOT USED.')
    name = path.split(f'.{extIn}')[0]

    # Don't automate because depends on what will be saving the JPG like OpenCV with its BGR format...
    #bgr2rgb = False
    #if extOut == 'jpg':
    #    bgr2rgb = True
    
    outDType = IMAGE_FORMAT_TO_BIT_DEPTH[extOut][0]
    img = readImage(path, extIn, outDType, bgr2rgb, rawpyGammaParams, rawpyNoAutobright, tonemap)

    if downsample:
        img = downscale(img, 4)

    savePath = f'{name}.{extOut}'
    writeImage(img, savePath, extOut)
    print(f'\tInput image path: {path}')
    print(f'\tOutput image path: {savePath}')
    return savePath





def testEXRPrecision():
    folder = f'/Users/countenance/Desktop'
    imgPathOrig = f'{folder}/TestEXR.exr'

    imgOrig = readImage(imgPathOrig, 'exr')
    printStats(imgOrig, 'init reading', filterNan=True)

    imgOrig = np.nan_to_num(imgOrig)
    imgOrig_f16 = imgOrig.astype(np.float16)
    imgOrig_f32 = imgOrig.astype(np.float32)

    # float16 looks the same as float32, so use that to halve the storage!
    writeImage(imgOrig_f16, f'{folder}/TestEXR-Out-Float16.exr')
    writeImage(imgOrig_f32, f'{folder}/TestEXR-Out-Float32.exr')

    return

def testPreprocessDataImages():
    from Paths import PATH_ONEDRIVE
    matName = 'ButterflySwallowtail'
    origPaths = sorted(glob.glob(f'{PATH_ONEDRIVE}/Retroreflection/128x1/{matName}/rectified/Retro_*.exr'))
    print(origPaths)
    
    cropBounds01 = None    # if None, then reads a CropBounds.py image in the rectified folder.

    #sessionName = 'Orange'
    sessionName = 'Black'
    #resizeRes = 150

    # TODO channels not interpreted correctly on reading a HalfEXR float16 image, resizing with skimage, and writing again to EXR (TODO check float16 here).
    resizeRes = None    # Turn off for now!  skimage doesn't work well with the EXRs.
    preprocessDataImages(origPaths, sessionName, cropBounds01, resizeRes, forceRewriteImages=True)

def testResizeAndWriteEXR():
   
    folder = f'{PATH_DOME_CONTROL}/DataOnRepo'
    image = readImage(f'{folder}/Tiny.exr')
    newShape = (100, 100, 3)
    image = skt.resize_local_mean(image, newShape)
    image = image.reshape(newShape)  # Force shape again
    writeImage(image, f'{folder}/TinyResize.exr')
    
    return

