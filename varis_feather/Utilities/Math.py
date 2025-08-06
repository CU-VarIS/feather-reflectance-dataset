# ----------------------------------------------------
#
# Math.py
#
# Utility data and functions concerning converting between hardware IDs and coordinate systems for
# material measurement, linear algebra functions, statistical algorithms...
#
# Last update: March 2024
# Last author: Jessica Baron
#
# ----------------------------------------------------


import numpy as np
np.set_printoptions(suppress=True)


INVALID_BRDF_FLAG = -1    # TODO Does numpy have a value like np.inf that could be more obvious?

PI = np.pi
PI_HALF = PI * 0.5
PI_2 = PI * 2.0
PI_3_HALF = PI_HALF * 3.0

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])
Z_AXIS = np.array([0.0, 0.0, 1.0])

# Dome spherical coordinate system:
#    North pole                 theta = 0
#    South pole                 theta = 180
#    Equator                    theta = 90
#    Prime meridian             phi = 0


def printStats(arr, name, filterNan=False):
    sizeInGB = arr.nbytes / 1e9
    numNaNs = np.isnan(arr).sum()
    numElements = np.prod(arr.shape)
    percent = (numNaNs / numElements) * 100
    prefix = (f'\tArray "{name}" \t shape={arr.shape}, dtype={arr.dtype}, '
              f'GB={sizeInGB:.4f}, numNaNs={numNaNs}/{numElements} ({percent:.2f}%)')
    if filterNan:
        print(f'\t{prefix} \n\t\t stats (NaNs ignored): \t'
              f'\t min={np.nanmin(arr):.4f} \t max={np.nanmax(arr):.4f} \t mean={np.nanmean(arr):.4f} '
              f'\t std={np.nanstd(arr):.4f} \t sum={np.nansum(arr):.4f}')
    else:
        print(f'\t{prefix} \n\t\t stats: \t\t'
              f'\t min={np.min(arr):.4f} \t max={np.max(arr):.4f} \t mean={np.mean(arr):.4f} '
              f'\t std={np.std(arr):.4f} \t sum={np.sum(arr):.4f}')


# https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
def rejectOutliers(data, m=2.0):
    diff = np.abs(data - np.median(data))
    med = np.median(diff)
    if med:
        s = diff / med
    else:
        s = np.zeros(len(diff))
    return data[s < m]


# Convenience function for vectorized dot product with einsum.
def dot_v(vectorsA, vectorsB):
    return np.einsum('...ij,...ij->...i', vectorsA, vectorsB)

def dot(vectorA, vectorB):
    #return np.sum(np.prod)
    dims = len(vectorA)
    total = 0
    for d in range(dims):
        total += vectorA[d] * vectorB[d]
    return total


def findPerpendicularVector(vector):
    # Pick an axis (1,0,0) or (0,1,0) to pair with given vector.
    # Take the cross between vector and axis.
    xAxis = np.array([1.0, 0.0, 0.0])
    if vector[0] == xAxis[0] and vector[1] == xAxis[1]:
        return np.cross(vector, np.array([0.0, 1.0, 0.0]))
    return np.cross(vector, xAxis)


#def magnitude_v(vecs):
#    x, y, z = vecs[:, 0], vecs[:, 1], vecs[:, 2]
#    return x*x + y*y + z*z


# Dot product (sum of products) to compute Euclidean distance.
# singleValue has shape (1, m) or (m, ). otherValues has shape (n, m).
def distance(singleValue, otherValues):
    if otherValues.ndim == 1:
        diff = otherValues - singleValue
        return np.sqrt(dot(diff, diff))

    else:
        n, m = otherValues.shape
        singleRepeat = np.broadcast_to(singleValue, (n, m))
        diff = otherValues - singleRepeat
        return np.sqrt(dot_v(diff, diff))


# From pbrt-v4
class PiecewiseLinear2D:

    def __init__(self, data2D):
        self.data = data2D
        self.sizeY, self.sizeX = data2D.shape
        self.sizeSlice = self.sizeX * self.sizeY
        self.PDF = None
        self.marginalCDF = None
        self.conditionalCDF = None

        self._buildCDF()


    def _buildCDF(self):
        dataFlat = self.data.ravel()
        self.marginalCDF = np.zeros(self.sizeY, dtype=np.float32)   # Marginal only on first dimension.
        self.conditionalCDF = np.zeros((self.sizeY, self.sizeX), dtype=np.float32)

        # Construct conditional CDF.
        total = 0.0
        for y in range(self.sizeY):
            for x in range(self.sizeX - 1):
                d0 = self.data[y, x]
                d1 = self.data[y, x + 1]
                total += 0.5 * (d0 + d1)
                self.conditionalCDF[y, x + 1] = total

        # Construct marginal CDF.
        total = 0.0
        x = self.sizeX - 1   # Last element of each row.
        for y in range(self.sizeY - 1):
            cCDF0 = self.conditionalCDF[y, x]
            cCDF1 = self.conditionalCDF[y + 1, x]
            total += 0.5 * (cCDF0 + cCDF1)
            self.marginalCDF[y + 1] = total

        # Normalize the CDFs and data as the PDF.
        norm = 1.0 / self.marginalCDF[-1]
        print('cond CDF shape', self.conditionalCDF.shape)
        print('cond CDF mn, mx, sum', self.conditionalCDF.min(), self.conditionalCDF.max(), np.sum(self.conditionalCDF))
        print('cond CDF data', self.conditionalCDF)
        print('marg CDF shape', self.marginalCDF.shape)
        print('marg CDF mn, mx, sum', self.marginalCDF.min(), self.marginalCDF.max(), np.sum(self.marginalCDF))
        print('marg CDF data', self.marginalCDF)
        print('data shape', self.data.shape)
        print('data mn, mx, sum', self.data.min(), self.data.max(), np.sum(self.data))

        self.conditionalCDF *= norm
        self.marginalCDF *= norm
        #self.PDF = self.data * integral
        integral = np.sum(self.data) # / np.prod(self.data.shape)
        self.PDF = self.data / integral
        print()
        print('marg CDF norm', norm)
        print('sum integral', integral)
        print('cond CDF mn, mx, sum', self.conditionalCDF.min(), self.conditionalCDF.max(), np.sum(self.conditionalCDF))
        print('marg CDF mn, mx, sum', self.marginalCDF.min(), self.marginalCDF.max(), np.sum(self.marginalCDF))
        print('PDF shape', self.PDF.shape)
        print('PDF mn, mx, sum', self.PDF.min(), self.PDF.max(), np.sum(self.PDF))
        print()


def movingAverage(arr, windowSize):
    window = np.ones(int(windowSize)) / float(windowSize)
    return np.convolve(arr, window, 'same')

# TODO improve endpoints more?
def movingAverageWithBufferAndDims(arr, windowSize, endBuffer, numDims):
    arrTmp = np.zeros((arr.shape[0] + endBuffer, numDims))  # Copying so first 'buf' values are duplicated.
    topN = int(0.1 * arr.shape[0])
    arrTmp[0:endBuffer] = np.average(arr[0:topN])
    arrTmp[endBuffer:] = arr.copy()
    for d in range(numDims):
        arrTmp[:, d] = movingAverage(arrTmp[:, d], windowSize)
    return arrTmp[endBuffer:].copy()


# Perfect (mirror) specular reflection of direction d about normal n.
def reflect(d, n):
    return 2.0 * np.dot(n, d) * n - d

# Vector version
def reflect_v(d, n):
    #return 2.0 * np.einsum('ij, ij->i', n, d) * n - d
    dot = np.sum(n*d, axis=1)
    return 2.0 * np.einsum('i, ij->ij', dot, n) - d


def halfway(v0, v1):
    return normalize(v0 + v1)
    
def halfway_v(v0, v1):
    return normalize_v(v0 + v1)


# vals0, vals1 should be the same data shape.
def grid(vals0, vals1, returnGrids, transposeGrids):
    # Fine for 1D data but confusing for more dims?
    if vals0.ndim != 1 or vals1.ndim != 1:
        print('WARNING: Math.grid() requires 1D arrays.')
        return None
    
    v0Grid, v1Grid = np.meshgrid(vals0, vals1)
    #v0Grid = v0Grid.ravel()
    #v1Grid = v1Grid.ravel()
    grid = np.column_stack([v0Grid.ravel(), v1Grid.ravel()])

    if returnGrids:
        if transposeGrids:
            return grid, v0Grid.T, v1Grid.T
        else:
            return grid.T, v0Grid, v1Grid

    else:
        if transposeGrids:
            return grid.T
        else:
            return grid


# Version with Z as world up.
def sphToCartZ(sph):
    sinTheta = np.sin(sph[0])
    w = np.array([sinTheta * np.cos(sph[1]), sinTheta * np.sin(sph[1]), np.cos(sph[0])])
    return normalize(w)

def sphToCartZ_v(sph):
    sinTheta = np.sin(sph[:, 0])
    w = np.column_stack([ sinTheta * np.cos(sph[:, 1]),
                          sinTheta * np.sin(sph[:, 1]),
                          np.cos(sph[:, 0]) ])
    return normalize_v(w)

def cartZToSph(dir):
    theta = np.arccos(dir[2])
    phi = np.arctan2(dir[1], dir[0])
    return np.array([theta, phi])

def cartZToSph_v(dirs):
    theta = np.arccos(dirs[:, 2])
    phi = np.arctan2(dirs[:, 1], dirs[:, 0])
    return np.column_stack([theta, phi])



# Parameters given in degrees.
# By 5 degrees seems to work for multiple applications...  Sampling lights for gradient pattern by 1 deg changes is way too fine.
def generatePolarCoords(useRadians, thetaStart=0, thetaEnd=180, thetaStep=5,
                                    phiStart=0, phiEnd=360, phiStep=10):
    t = np.arange(thetaStart, thetaEnd, thetaStep, dtype=np.float32)
    p = np.arange(phiStart, phiEnd, phiStep, dtype=np.float32)
    thetaMesh, phiMesh = np.meshgrid(t, p)
    sph = np.array([thetaMesh.ravel(), phiMesh.ravel()]).T
    if useRadians:
        return np.deg2rad(sph)
    return sph


def generateSpherePoints(cartesian, domeSpace=False,
                         thetaStart=0, thetaEnd=180, thetaStep=5,
                         phiStart=0, phiEnd=360, phiStep=10):

    sph = generatePolarCoords(True,   thetaStart, thetaEnd, thetaStep,   phiStart, phiEnd, phiStep)
    xyz = spherical_to_cartesian_v(sph, 1.0, True)

    if domeSpace:
        xyz = sample_to_dome_v(xyz, cartesian=True)
        sph = cartesian_to_spherical_v(xyz, 1.0, True)

    if cartesian:
        return xyz
    else:
        return sph


def magnitude_v(vectors):
    if len(vectors.shape) == 1:
        v = vectors
        return np.sqrt(np.dot(v, v))
    else:
        mag = np.zeros(vectors.shape[0])
        for i, v in enumerate(vectors):
            mag[i] = np.sqrt(np.dot(v, v))
        return mag


def normalize(vector):
    return vector / np.sqrt(np.dot(vector, vector))


def normalize_v(vectors):
    #if len(vectors.shape) == 1:
    #    v = vectors
    #    return v[..., :] / np.sqrt(np.dot(v[..., :], v[..., :]))
    #else:
    newV = np.zeros(vectors.shape)
    for i, v in enumerate(vectors):
        newV[i] = v / np.sqrt(np.dot(v, v))
    return newV


def getMatrixTransform2D(radians, transX, transY):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[c,  -s,  transX],
                  [s,   c,  transY],
                  [0.0, 0.0, 1.0]])
    return M


def getMatrixRotateX(radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return M

def rotateX(vector, radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[1, 0, 0],
                  [0, c, -s],
                  [0, s, c]])
    return np.dot(vector, M)

def rotateX_v(vectors, radians):
    N = len(vectors)
    ones, zeros = np.ones(N), np.zeros(N)
    try:      # Given a float
        c = np.broadcast_to(np.cos(radians), N)
        s = np.broadcast_to(np.sin(radians), N)
    except:   # Given an array.
        c, s = np.cos(radians), np.sin(radians)
    M = np.zeros((N, 3, 3))
    M[:, 0] = np.array([1, 0, 0])
    M[:, 1] = np.column_stack([zeros, c, -s])
    M[:, 2] = np.column_stack([zeros, s, c])
    v = vectors[:, None]
    newVectors = np.einsum('...cr,...rc->...c', v, M)
    #print('first v*M', np.dot(v[0], M[0]))
    #print('first new vec', newVectors[0])
    return newVectors


def getMatrixRotateY(radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return M

def rotateY(vector, radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[c, 0, s],
                  [0, 1, 0],
                  [-s, 0, c]])
    return np.dot(vector, M)

def rotateY_v(vectors, radians):
    N = len(vectors)
    ones, zeros = np.ones(N), np.zeros(N)
    try:
        c = np.broadcast_to(np.cos(radians), N)
        s = np.broadcast_to(np.sin(radians), N)
    except:
        c, s = np.cos(radians), np.sin(radians)
    s = np.broadcast_to(np.sin(radians), N)
    M = np.zeros((N, 3, 3))
    M[:, 0] = np.column_stack([c, zeros, s])
    M[:, 1] = np.array([0, 1, 0])
    M[:, 2] = np.column_stack([-s, zeros, c])
    v = vectors[:, None]
    newVectors = np.einsum('...cr,...rc->...c', v, M)
    return newVectors


def getMatrixRotateZ(radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[c, -s, 0],
                  [s, c, 0],
                  [0, 0, 1]])
    return M

def rotateZ(vector, radians):
    c, s = np.cos(radians), np.sin(radians)
    M = np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])
    return np.dot(vector, M)

def rotateZ_v(vectors, radians):
    N = len(vectors)
    ones, zeros = np.ones(N), np.zeros(N)
    try:
        c = np.broadcast_to(np.cos(radians), N)
        s = np.broadcast_to(np.sin(radians), N)
    except:
        c, s = np.cos(radians), np.sin(radians)
    s = np.broadcast_to(np.sin(radians), N)
    M = np.zeros((N, 3, 3))
    M[:, 0] = np.column_stack([c, -s, zeros])
    M[:, 1] = np.column_stack([s, c, zeros])
    M[:, 2] =  np.array([0, 0, 1])
    v = vectors[:, None]
    newVectors = np.einsum('...cr,...rc->...c', v, M)
    return newVectors


def getRotateAngleAxisMatrix(radians, axis):
    if np.isclose(radians, 0.0):
        return np.identity(3)

    half = radians * 0.5
    a = np.cos(half)
    b, c, d = -axis * np.sin(half)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    M = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    return M


def rotateAngleAxis(vector, radians, axis, quaternion=False):
    if quaternion:
        #quat0 = Quaternion(0.0, axis)
        quat1 = Quaternion(radians, axis)
        M = quat1.toMatrix()
        v = np.array([vector[0], vector[1], vector[2], 0.0])
        return np.dot(v, M)[:3]

    else:
        M = getRotateAngleAxisMatrix(radians, axis)
        return np.dot(vector, M)


def rotateAngleAxis_v(vectors, radians, axes, quaternion=True, getMatrix=False):
    if quaternion:
        newVectors = []
        for i in range(len(vectors)):
            v = rotateAngleAxis(vectors[i], radians[i], axes[i], True)
            newVectors.append(v)
        return np.array(newVectors)

    else:
        N = len(vectors)
        half = radians * 0.5
        a = np.cos(half)
        s = -axes * np.sin(half)[:, None]
        b, c, d = s[:, 0], s[:, 1], s[:, 2]
        aa, bb, cc, dd = a * a, b * b, c * c, d * d
        bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
        M = np.zeros((N, 3, 3))
        M[:, 0] = np.column_stack([aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)])
        M[:, 1] = np.column_stack([2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)])
        M[:, 2] = np.column_stack([2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc])
        v = vectors[:, None]
        newVectors = np.einsum('...cr,...rc->...c', v, M)
        newVectors = setNearValueToValue(newVectors, value=0.0, thresh=0.001)
        newVectors = setNearValueToValue(newVectors, value=1.0, thresh=0.001)
        newVectors = normalize_v(newVectors)
        if getMatrix:
            return newVectors, M
        else:
            return newVectors


def testRotQuat():
    v = np.array([1,0,0])
    ax = normalize_v(np.array([0,1,1]))
    rads = PI_HALF
    vNewQ = rotateAngleAxis(v, rads, ax, quaternion=True)
    vNewM = rotateAngleAxis(v, rads, ax, quaternion=False)
    print(f'Result with matrix: {vNewM}')
    print(f'Result with quat: {vNewQ}')
#testRotQuat()


def setNearValueToValue(arr, value, thresh=0.0001):
    inds = np.isclose(a=arr, b=value, atol=thresh)
    arr[inds] = value
    return arr


def getCoordinateFrame(v, handedness='right', dtype=np.float32):
    #v1 = normalize_v(getOrthogonalVector(v, v))
    #if handedness == 'right':
    #    v2 = normalize_v(getOrthogonalVector(v1, v))
    #else:
    #    v2 = normalize_v(getOrthogonalVector(v, v1))
    #frame = np.array([v2, v1, v], dtype=dtype)

    # sph coords in respect to dome Y axis ((0,0) sph is (0,1,0) xyz).
    # theta about dome Z (when phi==0), phi about Y
    # theta about Z changed to theta about X
    # phi still about Y but shifted to start from +Z and not +X
    sphZ = cartesian_to_spherical_v(v, 1.0, useRadians=True).ravel()
    cartZ = spherical_to_cartesian_v(sphZ, 1.0, useRadians=True).ravel()

    # TODO work with spherical space?
    halfhalf = PI_HALF * 0.5
    #sphY = np.array([sphZ[0] + PI_HALF,  sphZ[1] + PI_HALF])
    sphY = np.array([sphZ[0] + PI_HALF, sphZ[1] + PI])
    cartY = spherical_to_cartesian_v(sphY, 1.0, useRadians=True).ravel()

    #sphX = np.array([sphY[0] + PI_HALF,  sphY[1] + PI_HALF])
    #sphX = np.array([sphZ[0] + PI, sphZ[1] + PI_HALF])
    #cartX = spherical_to_cartesian_v(sphX, 1.0, useRadians=True).ravel()

    b = getOrthogonalVector(v, cartY)
    t = getOrthogonalVector(b, v)


    #theta = PI_HALF + sph[0]
    #phi = -(PI_HALF + sph[1])

    #thetaMod = putAngleInRange(theta, -PI, PI)
    #phiMod = putAngleInRange(phi, -PI_2, PI_2)
    #T0 = getMatrixRotateX(thetaMod)
    #T1 = getMatrixRotateY(phiMod)

    #T0 = getMatrixRotateZ(sph[0])
    #T1 = getMatrixRotateY(-sph[1])
    #Rx = getMatrixRotateX(-PI_HALF)
    #Ry = getMatrixRotateY(PI_HALF)
    #T = Ry @ Rx @ T1 @ T0    # theta first because phi first does nothing at Y axis

    # Shows that T transforms original vector v to the Z axis 0,0,1.

    #vXZ = normalize_v(np.array([v[0], v[2]]))
    #zXZ = np.array([0, 1])
    #thetaSS = np.arccos(np.dot(vXZ, zXZ))
    #T0 = getMatrixRotateY(-thetaSS)
    #vTmp = T0 @ v
    #if np.isclose(np.dot(Z_AXIS, vTmp), 1.0):
    #    phiSS = 0.0
    #    T1 = np.identity(3)
    #else:
    #    vTmpXY = normalize_v(np.array([vTmp[0], vTmp[1]]))
     ##   yXY = np.array([1, 0])
     #   phiSS = np.arccos(np.dot(vTmpXY, yXY))
     #   phiSS = np.mod(phiSS, PI)
     #   T1 = getMatrixRotateZ(-phiSS)


    # Transform X and Y axes to tangent t and bitangent b.
    #t = T @ X_AXIS
    #b = T @ Y_AXIS

    if handedness == 'right':
        TBN = np.array([t, b, v])
    else:
        TBN = np.array([b, t, v])

    TBN = setNearValueToValue(TBN, value=0.0, thresh=0.0001)
    TBN = setNearValueToValue(TBN, value=1.0, thresh=0.0001)
    TBN = normalize_v(TBN)
    #testCoordinateFrame(TBN[0], TBN[1], TBN[2])
    return TBN


def testCoordinateFrame(v0, v1, v2):
    print('testCoordinateFrame(): All dot products should be 0.')
    print('\tv0 dot v1', np.dot(v0, v1))
    print('\tv1 dot v0', np.dot(v1, v0))
    print('\tv0 dot v2', np.dot(v0, v2))
    print('\tv2 dot v0', np.dot(v2, v0))
    print('\tv1 dot v2', np.dot(v1, v2))
    print('\tv2 dot v1', np.dot(v2, v1))


def matrixToEulerAngles(M, getRadians, version=0):
    order = None

    # According to 8.7.2 in Dunn and Parberry Math Primer book. YXZ (HPB) order?
    # "heading, pitch, bank" (yaw, pitch, roll)
    if version == 0:
        p = -M[2, 1]
        if p <= -1.0:
            pitch = -PI_HALF
        elif p >= 1.0:
            pitch = PI_HALF
        else:
            pitch = np.arcsin(p)

        # Gimbal-lock check.
        if np.fabs(pitch) > 0.9999:
            yaw = np.arctan2(-M[0, 2], M[0, 0])
            roll = 0.0
        else:
            yaw = np.arctan2(M[2, 0], M[2, 2])
            roll = np.arctan2(M[0, 1], M[1, 1])
        angles = np.array([yaw, pitch, roll]).astype(M.dtype)
        order = 'YXZ'

    # XYZ order in Mike Day's Euler Angles https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2012/07/euler-angles.pdf
    elif version == 1:
        pitch = np.arctan2(M[1, 2], M[2, 2])
        c2 = np.sqrt(M[0, 0] * M[0, 0] + M[0, 1] * M[0, 1])
        yaw = np.arctan2(-M[0, 2], c2)
        c1, s1 = np.cos(pitch), np.sin(pitch)
        roll = np.arctan2(s1 * M[2, 0] - c1 * M[1, 0],
                          c1 * M[1, 1] - s1 * M[2, 1])
        angles = np.array([pitch, yaw, roll]).astype(M.dtype)
        order = 'XYZ'

    # ZXY order in sec 2.5 here https://www.geometrictools.com/Documentation/EulerAngles.pdf
    elif version == 2:
        if M[2, 1] < 1.0:
            if M[2, 1] > -1.0:
                pitch = np.arcsin(M[2, 1])
                roll = np.arctan2(-M[0, 1], M[1, 1])
                yaw = np.arctan2(-M[2, 0], M[2, 2])
            else:  # M[2,1] == -1
                pitch = -PI_HALF
                roll = -np.arctan2(M[0, 2], M[0, 0])
                yaw = 0.0
        else:   # M[2,1] == 1.0
            pitch = PI_HALF
            roll = np.arctan2(M[0, 2], M[0, 0])
            yaw = 0.0
        angles = np.array([roll, yaw, pitch]).astype(M.dtype)
        order = 'ZYX'

    # YXZ order in sec 2.3 here https://www.geometrictools.com/Documentation/EulerAngles.pdf
    # !!! May be best for dome motors starting from motor2 (Z) then middle motor1 (X) then outermost motor0 (Y).
    elif version == 3:
        if M[1, 2] < 1.0:
            if M[1, 2] > -1.0:
                pitch = np.arcsin(-M[1, 2])
                yaw = np.arctan2(M[0, 2], M[2, 2])
                roll = np.arctan2(M[1, 0], M[1, 1])
            else:
                pitch = PI_HALF
                yaw = -np.arctan2(-M[0, 1], M[0, 0])
                roll = 0.0
        else:
            pitch = -PI_HALF
            yaw = np.arctan2(-M[0, 1], M[0, 0])
            roll = 0.0
        angles = np.array([yaw, pitch, roll]).astype(M.dtype)
        order = 'YXZ'

    if getRadians:
        return angles, order
    else:
        return np.rad2deg(angles), order


# Cross product of parallel vectors is 0.
def getOrthogonalVector(v0, v1):
    cross = np.cross(v0, v1)
    # Parallel vectors so do a dot-product approach instead.
    if np.isclose(np.sum(np.abs(cross)), 0):
        # Orthogonal axis v2 will have a dot product of 0 with BOTH vectors which are parallel, so solve for one.
        # Fix some x2,y2 and solve for z2 (or solving for whichever coord is nonzero).
        #     x0*x2 + y0*y2 + z0*z2 = 0
        #     -(z0*z2) = x0*x2 + y0*y2
        #     z2 = (x0*x2 + y0*y2) / -z0
        x, y, z = v0
        if not np.isclose(z, 0):     # Can divide by v0's Z coord.
            x2 = y2 = 1.0
            z2 = (x2*x + y2*y) / -z
        elif not np.isclose(y, 0):   # Try Y.
            x2 = z2 = 1.0
            y2 = (z2*z + x2*x) / -y
        elif not np.isclose(x, 0):   # Try X.
            y2 = z2 = 1.0
            x2 = (z2*z + y2*y) / -x
        else:
            return None
        axis = normalize_v(np.array([x2, y2, z2]))
    else:
        axis = normalize_v(cross)
    return axis


def angleAxisRotationToEulerAngles(vector0, vector1):
    angle = np.arccos(np.dot(vector0, vector1))
    if np.isclose(angle, 0.0):
        return np.zeros(3)
    else:
        axis = getOrthogonalVector(vector0, vector1)
        M = getRotateAngleAxisMatrix(angle, axis)
        return matrixToEulerAngles(M)


# According to 8.7.2 in Dunn and Parberry Math Primer book.
# Z, X, Y order
def eulerAnglesToVector(startVector, eulerDeg, version=3):
    euler = np.deg2rad(eulerDeg)
    pitch, yaw, roll = euler

    if version == 0:
        vec = rotateZ(startVector, roll)
        vec = rotateX(vec, pitch)
        vec = rotateY(vec, yaw)
        return vec

    elif version == 1:   # same as version 0
        ch, sh = np.cos(-yaw), np.sin(-yaw)
        cp, sp = np.cos(-pitch), np.sin(-pitch)
        cb, sb = np.cos(-roll), np.sin(-roll)
        M = np.array([ [ch*cb + sh*sp*sb,   -ch*sb + sh*sp*cb,  sh*cp],
                       [sb*cp,              cb*cp,              -sp],
                       [-sh*cb + ch*sp*sb,  sb*sh + ch*sp*cb,   ch*cp] ])
        return np.dot(M, startVector)

    elif version == 2:
        vec = rotateY(startVector, yaw)
        vec = rotateX(vec, pitch)
        vec = rotateZ(vec, roll)
        return vec

    elif version == 3:   # same as version 2
        ch, sh = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cb, sb = np.cos(roll), np.sin(roll)
        M = np.array([ [ch*cb + sh*sp*sb,   sb*cp,   -sh*cb + ch*sp*sb],
                       [-ch*sb + sh*sp*cb,  cb*cp,   sb*sh + ch*sp*cb],
                       [sh*cp,              -sp,     ch*cp] ])
        return np.dot(M, startVector)


def testEulerVectorConversion():
    #v0 = normalize_v(np.array([1, 1, 1]))
    v0 = normalize_v(np.array([1, 0, 0]))
    xs = np.arange(-1.0, 1.0, step=0.2)
    ys = np.arange(-1.0, 1.0, step=0.2)
    for x in xs:
        for y in ys:
            v1 = normalize_v(np.array([x, y, 0.1]))
            euler = angleAxisRotationToEulerAngles(v0, v1)
            res0 = eulerAnglesToVector(v0, euler, version=0)
            res1 = eulerAnglesToVector(v0, euler, version=1)
            res2 = eulerAnglesToVector(v0, euler, version=2)
            res3 = eulerAnglesToVector(v0, euler, version=3)
            print('\nv0', v0)
            print('v1 (goal)', v1)
            print('euler', euler)
            print(f'result 0: {res0}, dot with v1 = {np.dot(v1, res0)}')
            print(f'result 1: {res1}, dot with v1 = {np.dot(v1, res1)}')
            print(f'result 2: {res2}, dot with v1 = {np.dot(v1, res2)}')
            print(f'result 3: {res3}, dot with v1 = {np.dot(v1, res3)}')
#testEulerVectorConversion()


# Change polar coords by Euler-ish angles.
def rotateByPolar(cartesian, dTheta, dPhi):
    sph = cartesian_to_spherical_v(cartesian, 1, True)
    sph[:, 0] += dTheta
    sph[:, 1] += dPhi
    return spherical_to_cartesian_v(sph, 1, True)


# Max is typically 2pi or pi radians. Min WAS implicitly 0.
def putAngleInRange_v(angles, minAngle, maxAngle):
    minRepeat = np.broadcast_to(minAngle, angles.shape).astype(np.float32)
    maxRepeat = np.broadcast_to(maxAngle, angles.shape).astype(np.float32)
    range = maxAngle - minAngle

    indsNeg = np.argwhere(angles < 0.0).ravel()
    indsPos = np.argwhere(angles >= 0.0).ravel()
    if len(indsNeg) > 0:
        #angles[indsNeg] = minRepeat[indsNeg] - np.mod(angles[indsNeg], range)
        angles[indsNeg] = minRepeat[indsNeg] + np.mod(angles[indsNeg], range)
    if len(indsPos) > 0:
        angles[indsPos] = minRepeat[indsPos] + np.mod(angles[indsPos], range)

    return angles


def putAngleInRange(angle, minAngle, maxAngle):
    if minAngle <= angle and angle <= maxAngle:
        return angle

    range = maxAngle - minAngle
    if angle < 0.0:
        #angle = minAngle - np.mod(angle, range)
        angle = minAngle + np.mod(angle, range)
    else:
        angle = minAngle + np.mod(angle, range)
        #angle = maxAngle - np.mod(angle, range)
    return float(angle)    # np.mod() makes a weird array type?


# **NEW** FOR BRDFs, undo domeToSample_v.
# Testing with spherical_to_lightId_v().
def sampleToDome_v(coords, useCartesian):
    if useCartesian:
        xyz = coords
    else:
        xyz = sphToCartZ_v(coords)
    xyz = rotateZ_v(xyz, -PI)
    xyz = rotateX_v(xyz, -PI_HALF)
    if useCartesian:
        return xyz
    else:
        return cartZToSph_v(xyz)


# Dome spherical coordinates --> Sample/object spherical coordinates
# **NEW** FOR BRDFs, sample surface should be pointing so its normal is at dome coords (90, 90) +X axis (towards inside room).
def domeToSample_v(coords, useCartesian):
    if useCartesian:
        xyz = coords
    else:
        xyz = sphToCartZ_v(coords)
   
    # close
    xyz = rotateY_v(xyz, PI_HALF)
    #xyz = rotateZ_v(xyz, PI)
    xyz = rotateX_v(xyz, PI)

    if useCartesian:
        return xyz
    else:
        return cartZToSph_v(xyz)



# Spherical : Cartesian
# [90, 0]   : [1, 0, 0]
# [0, 0]    : [0, 1, 0]
# [90, 90]  : [0, 0, 1]
# [90, 180] : [-1, 0, 0]
# [180, 0]  : [0, -1, 0]
# [90, 270] : [0, 0, -1]
def spherical_to_cartesian_v(coords, radius, useRadians):
    if not useRadians:
        coords = np.deg2rad(coords)

    if len(coords.shape) == 2:   # if many coords
        theta, phi = coords[:, 0], coords[:, 1]
    else:                        # if single coord
        theta, phi = np.array([coords[0]]), np.array([coords[1]])

    cosTheta, sinTheta = np.cos(theta), np.sin(theta)
    cosPhi, sinPhi = np.cos(phi), np.sin(phi)
    x = radius * cosPhi * sinTheta
    y = radius * cosTheta
    z = radius * sinPhi * sinTheta
    xyz = np.column_stack([x, y, z])
    return xyz


def cartesian_to_spherical_v(coords, radius, useRadians):
    if len(coords.shape) == 2:   # if many coords
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    else:                        # if single coord
        x, y, z = coords[0], coords[1], coords[2]
        x, y, z = np.array([x]), np.array([y]), np.array([z])

    theta = np.arccos(y / radius)

    phi = np.arctan2(z, x) + PI_2
    phi = putAngleInRange_v(phi, 0.0, PI_2)

    # For no elevation, azimuth doesn't matter at all.
    i = np.where(theta == 0)
    phi[i] = 0

    # arctan2 seems to handle all of these manual conditions.
    #phi = -999 * np.ones(theta.shape, dtype=np.float32)
    #i = np.where(x > 0)
    #phi[i] = np.arctan(z[i] / x[i])
    #i = np.where((x < 0) & (z >= 0))
    #phi[i] = (np.arctan(z[i] / x[i]) + PI)
    #i = np.where((x < 0) & (z < 0))
    #phi[i] = (np.arctan(z[i] / x[i]) - PI)
    #i = np.where((x == 0) & (z > 0))
    #phi[i] = PI * 0.5
    #i = np.where((x == 0) & (z < 0))
    #phi[i] = -PI * 0.5
    #i = np.where(theta == 0)
    #phi[i] = 0.0

    if not useRadians:
        return np.rad2deg(np.column_stack([theta, phi]))
    return np.column_stack([theta, phi])


def test_spherical_and_cartesian():
    radius = 1.0
    rads = True

    sph = generatePolarCoords(True)
    xyz = spherical_to_cartesian_v(sph, radius, rads)
    sphRet = cartesian_to_spherical_v(xyz, radius, rads)
    print('Full sphere')
    print('polar', np.rad2deg(sph[0:-1:100]))
    print('cartesian', xyz[0:-1:100])
    print('polar returned', sphRet[0:-1:100])
    print('polar DIFF', sph[:10] - sphRet[:10])

    # Since no elevation change, should all be (0,1,0) in dome space.
    sph = generatePolarCoords(True,   0, 1, 1,   0, 360, 60)
    xyz = spherical_to_cartesian_v(sph, radius, rads)
    sphRet = cartesian_to_spherical_v(xyz, radius, rads)
    print('\nZero thetas')
    print('polar', sph[:10])
    print('cartesian', xyz[:10])
    print('polar returned', sphRet[:10])
    print('polar DIFF', sph[:10] - sphRet[:10])

    # Since no azimuth change, should all be in YZ plane in dome space.
    sph = generatePolarCoords(True,   0, 180, 30,   0, 1, 1)
    xyz = spherical_to_cartesian_v(sph, radius, rads)
    sphRet = cartesian_to_spherical_v(xyz, radius, rads)
    print('\nZero phis')
    print('polar', sph[:10])
    print('cartesian', xyz[:10])
    print('polar returned', sphRet[:10])
    print('polar DIFF', sph[:10] - sphRet[:10])

    # All values should be in postive quadrant.
    sph = generatePolarCoords(True,   0, 90, 30,   0, 90, 30)
    xyz = spherical_to_cartesian_v(sph, radius, rads)
    sphRet = cartesian_to_spherical_v(xyz, radius, rads)
    print('\nPositive quadrant')
    print('polar', sph[:10])
    print('cartesian', xyz[:10])
    print('polar returned', sphRet[:10])
    print('polar DIFF', sph[:10] - sphRet[:10])

    # And negative quadrant.
    sph = generatePolarCoords(True,   90, 180, 30,   180, 270, 30)
    xyz = spherical_to_cartesian_v(sph, radius, rads)
    sphRet = cartesian_to_spherical_v(xyz, radius, rads)
    print('\nNegative quadrant')
    print('polar', sph[:10])
    print('cartesian', xyz[:10])
    print('polar returned', sphRet[:10])
    print('polar DIFF', sph[:10] - sphRet[:10])
#test_spherical_and_cartesian()


# Main conversion equations (sec. 3.1) from Aghayari 2017 paper on Ricoh Theta Z1 (360 camera).
def pixel_to_dome_v(coords, imageWidth=1, imageHeight=1):
    if len(coords.shape) == 2:   # if many coords
        x, y = coords[:, 0], coords[:, 1]
    else:                        # if single coord
        x, y = np.array([coords[0]]), np.array([coords[1]])
    
    # Original based on paper.
    theta = (0.5 * imageHeight - y) * (PI / imageHeight)
    theta = PI_HALF - theta   # Shifting for dome

    u = np.modf(x / imageWidth + 0.5)[0]
    phi = u * PI_2
    # Original based on paper... Not best with VarIS setup?
    #phi = (x - 0.5 * imageWidth) * (PI_2 / imageWidth)
    #phi = putAngleInRange(phi + PI_HALF, PI_2)   # NOTE. Don't use this anymore. Use mod in 0..1 space.
    
    return np.column_stack([theta, phi])


# Reversing image_to_dome but in UV (0..1) space vs. an image of a particular size.
def dome_to_pixel_v(coords, imageWidth=1, imageHeight=1, useRadians=True):
    if not useRadians:
        coords = np.deg2rad(coords)

    if len(coords.shape) == 2:   # if many coords
        theta = coords[:, 0]
        phi = coords[:, 1]
    else:                        # if single coord
        theta = np.array([coords[0]])
        phi = np.array([coords[1]])

    # Need to rotate/mod u horizontally.
    # modf() takes fractional component (modding by 1.0).
    u = phi / PI_2
    u = np.modf(u + 0.5)[0]

    # For OLAT light coords test
    v = theta / PI
    #v = 1 - theta / PI
    #v = np.modf(v)[0]

    return np.column_stack([u * imageWidth,
                            v * imageHeight])


if __name__ == '__main__':
    v0 = np.array([1,0,0])
