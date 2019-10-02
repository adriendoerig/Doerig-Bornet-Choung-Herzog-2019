'''
Version: 2019-06-27 (Adrien & Oh-hyeon)
- added noise patch (with size//2)
'''

# Class to make a batch
import matplotlib
matplotlib.use('TkAgg')
import numpy, random, matplotlib.pyplot as plt
from skimage import draw
from scipy.ndimage import zoom
from datetime import datetime


random_pixels = 0  # stimulus pixels are drawn from random.uniform(1-random_pixels,1+random_pixels). So use 0 for deterministic shapes. Background noise is added independantly of this.

def all_test_shapes():
    return seven_shapesgen(5)+shapesgen(5)+Lynns_patterns()+ten_random_patterns()+Lynns_patterns()

def shapesgen(max, emptyvect=True):
    if max>7:
        return

    if emptyvect:
        s = [[]]
    else:
        s = []
    for i in range(1,max+1):
        s += [[i], [i,i,i], [i,i,i,i,i]]
        for j in range(1,max+1):
            if j != i:
                s += [[i,j,i,j,i]]

    return s

def seven_shapesgen(max, emptyvect=True):
    if max>7:
        return

    if emptyvect:
        s = [[]]
    else:
        s = []
    for i in range(1,max+1):
        s += [[i,i,i,i,i,i,i]]
        for j in range(1,max+1):
            if j != i:
                s += [[j,i,j,i,j,i,j]]

    return s

def Lynns_patterns():
    squares = [1, 1, 1, 1, 1, 1, 1]
    onesquare = [0, 0, 0, 1, 0, 0, 0]
    S = [squares]
    for x in [6,2]:
        line1 = [x,1,x,1,x,1,x]
        line2 = [1,x,1,x,1,x,1]

        line0 = [x,1,x,0,x,1,x]

        columns = [line1, line1, line1]
        checker = [line2, line1, line2]
        if x == 6:
            special = [1,x,2,x,1,x,1]
        else:
            special = [1,x,1,x,6,x,1]

        checker_special = [line2, line1, special]

        irreg = [[1,x,1,x,x,1,1], line1, [1,1,x,x,1,x,1]]
        cross = [onesquare, line1, onesquare]
        pompom = [line0, line1, line0]

        S +=[line1, columns, checker, irreg, pompom, cross, checker_special]
    return S

def ten_random_patterns(newone = False):
    patterns = numpy.zeros((10, 3, 7),dtype=int)
    if newone:
        basis = [0,1,2,6]
        for pat in range(10):
            for row in range(3):
                for col in range(7):
                    a = numpy.random.choice(basis)
                    patterns[pat][row][col] = a
    else:
        patterns = [[[6, 1, 1, 0, 1, 6, 2], [0, 1, 0, 1, 2, 1, 1], [1, 0, 1, 6, 6, 2, 6]],
                    [[1, 6, 1, 1, 2, 0, 2], [6, 2, 2, 6, 0, 1, 2], [1, 1, 0, 6, 1, 1, 1]],
                    [[1, 6, 1, 2, 2, 0, 2], [1, 0, 6, 1, 2, 2, 6], [2, 2, 0, 1, 0, 2, 1]],
                    [[6, 6, 0, 1, 1, 6, 6], [1, 1, 1, 2, 2, 6, 1], [6, 6, 2, 1, 6, 0, 6]],
                    [[0, 6, 2, 2, 2, 6, 6], [2, 0, 1, 1, 6, 6, 6], [1, 0, 6, 0, 2, 6, 2]],
                    [[2, 1, 1, 6, 2, 6, 2], [6, 1, 0, 6, 1, 2, 1], [1, 6, 0, 2, 1, 2, 6]],
                    [[1, 1, 0, 6, 6, 6, 1], [1, 0, 0, 1, 2, 1, 1], [2, 1, 0, 2, 6, 1, 6]],
                    [[0, 6, 6, 2, 2, 0, 2], [1, 6, 1, 6, 6, 2, 2], [2, 1, 6, 1, 0, 2, 2]],
                    [[6, 1, 2, 6, 1, 0, 1], [0, 1, 6, 2, 0, 6, 2], [1, 0, 1, 2, 6, 6, 6]],
                    [[1, 0, 1, 6, 2, 6, 2], [0, 6, 6, 2, 0, 1, 1], [6, 6, 1, 6, 0, 2, 1]]]
        return patterns


class StimMaker:

    def __init__(self, imSize, shapeSize, barWidth):

        self.imSize    = imSize
        self.shapeSize = shapeSize
        self.barWidth  = barWidth
        self.barHeight = int(shapeSize/4-barWidth/4)
        self.offsetHeight = 1


    def setShapeSize(self, shapeSize):

        self.shapeSize = shapeSize


    def drawSquare(self):

        resizeFactor = 1.2
        patch = numpy.zeros((self.shapeSize, self.shapeSize))

        firstRow = int((self.shapeSize - self.shapeSize/resizeFactor)/2)
        firstCol = firstRow
        sideSize = int(self.shapeSize/resizeFactor)

        patch[firstRow         :firstRow+self.barWidth,          firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow+sideSize:firstRow+self.barWidth+sideSize, firstCol:firstCol+sideSize+self.barWidth] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstCol         :firstCol+self.barWidth         ] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[firstRow:firstRow+sideSize+self.barWidth, firstRow+sideSize:firstRow+self.barWidth+sideSize] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawCircle(self):

        resizeFactor = 1.01
        radius = self.shapeSize/(2*resizeFactor)
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2)-1, int(self.shapeSize/2)-1) # due to discretization, you maybe need add or remove 1 to center coordinates to make it look nice

        for row in range(self.shapeSize):
            for col in range(self.shapeSize):

                distance = numpy.sqrt((row-center[0])**2 + (col-center[1])**2)
                if radius-self.barWidth < distance < radius:
                    patch[row, col] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawDiamond(self):
        S = self.shapeSize
        mid = int(S/2)
        resizeFactor = 1.00
        patch = numpy.zeros((S,S))
        for i in range(S):
            for j in range(S):
                if i == mid+j or i == mid-j or j == mid+i or j == 3*mid-i-1:
                    patch[i,j] = 1

        return patch


    def drawNoise(self):
        S = self.shapeSize
        patch = numpy.random.normal(0, .1, size=(S//3,S//3))
        return patch


    def drawPolygon(self, nSides, phi):

        resizeFactor = 1.0
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(nSides):
            rowExtVertices.append( radius               *numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
            colExtVertices.append( radius               *numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])
            rowIntVertices.append((radius-self.barWidth)*numpy.sin(2*numpy.pi*n/nSides + phi) + center[0])
            colIntVertices.append((radius-self.barWidth)*numpy.cos(2*numpy.pi*n/nSides + phi) + center[1])

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawStar(self, nTips, ratio, phi):

        resizeFactor = 0.8
        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        radius = self.shapeSize/(2*resizeFactor)

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        for n in range(2*nTips):

            thisRadius = radius
            if not n%2:
                thisRadius = radius/ratio

            rowExtVertices.append(max(min( thisRadius               *numpy.sin(2*numpy.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colExtVertices.append(max(min( thisRadius               *numpy.cos(2*numpy.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))
            rowIntVertices.append(max(min((thisRadius-self.barWidth)*numpy.sin(2*numpy.pi*n/(2*nTips) + phi) + center[0], self.shapeSize), 0.0))
            colIntVertices.append(max(min((thisRadius-self.barWidth)*numpy.cos(2*numpy.pi*n/(2*nTips) + phi) + center[1], self.shapeSize), 0.0))

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        return patch


    def drawIrreg(self, nSidesRough, repeatShape):

        if repeatShape:
            random.seed(1)

        patch  = numpy.zeros((self.shapeSize, self.shapeSize))
        center = (int(self.shapeSize/2), int(self.shapeSize/2))
        angle  = 0  # first vertex is at angle 0

        rowExtVertices = []
        colExtVertices = []
        rowIntVertices = []
        colIntVertices = []
        while angle < 2*numpy.pi:

            if numpy.pi/4 < angle < 3*numpy.pi/4 or 5*numpy.pi/4 < angle < 7*numpy.pi/4:
                radius = (random.random()+2.0)/3.0*self.shapeSize/2
            else:
                radius = (random.random()+1.0)/2.0*self.shapeSize/2

            rowExtVertices.append( radius               *numpy.sin(angle) + center[0])
            colExtVertices.append( radius               *numpy.cos(angle) + center[1])
            rowIntVertices.append((radius-self.barWidth)*numpy.sin(angle) + center[0])
            colIntVertices.append((radius-self.barWidth)*numpy.cos(angle) + center[1])

            angle += (random.random()+0.5)*(2*numpy.pi/nSidesRough)

        RR, CC = draw.polygon(rowExtVertices, colExtVertices)
        rr, cc = draw.polygon(rowIntVertices, colIntVertices)
        patch[RR, CC] = random.uniform(1-random_pixels, 1+random_pixels)
        patch[rr, cc] = 0.0

        if repeatShape:
            random.seed(datetime.now())

        return patch


    def drawStuff(self, nLines):

        patch  = numpy.zeros((self.shapeSize, self.shapeSize))

        for n in range(nLines):

            (r1, c1, r2, c2) = numpy.random.randint(self.shapeSize, size=4)
            rr, cc = draw.line(r1, c1, r2, c2)
            patch[rr, cc] = random.uniform(1-random_pixels, 1+random_pixels)

        return patch


    def drawVernier(self, offset=None, offset_size=None):

        if offset_size is None:
            offset_size = random.randint(1, int(self.barHeight/2.0))
        patch = numpy.zeros((2*self.barHeight+self.offsetHeight, 2*self.barWidth+offset_size))
        patch[0:self.barHeight, 0:self.barWidth] = 1.0
        patch[self.barHeight+self.offsetHeight:, self.barWidth+offset_size:] = random.uniform(1-random_pixels, 1+random_pixels)

        if offset is None:
            if random.randint(0, 1):
                patch = numpy.fliplr(patch)
        elif offset == 1:
            patch = numpy.fliplr(patch)

        fullPatch = numpy.zeros((self.shapeSize, self.shapeSize))
        firstRow  = int((self.shapeSize-patch.shape[0])/2)
        firstCol  = int((self.shapeSize-patch.shape[1])/2)
        fullPatch[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        return fullPatch


    def drawShape(self, shapeID, offset=None, offset_size=None):

        if shapeID == 0:
            patch = numpy.zeros((self.shapeSize, self.shapeSize))
        if shapeID == 1:
            patch = self.drawSquare()
        if shapeID == 2:
            patch = self.drawCircle()
        if shapeID == 3:
            patch = self.drawPolygon(6, 0)
        if shapeID == 4:
            patch = self.drawPolygon(8, numpy.pi/8)
        if shapeID == 5:
            patch = self.drawDiamond()
        if shapeID == 6:
            patch = self.drawStar(7, 1.7, -numpy.pi/14)
        if shapeID == 7:
            patch = self.drawIrreg(15, False)
        if shapeID == 8:
            patch = self.drawIrreg(15, True)
        if shapeID == 9:
            patch = self.drawStuff(5)
        if shapeID == 10:
            patch = self.drawNoise()

        return patch


    def drawStim(self, vernier_ext, shapeMatrix, vernier_in=False, offset=None, offset_size=None, fixed_position=None, noise_patch=None):
        if shapeMatrix == None:
            ID = numpy.random.randint(1, 7)
            siz = numpy.random.randint(4)*2 +1
            h = numpy.random.randint(2)*2 +1
            shapeMatrix = numpy.zeros((h,siz)) + ID

        image        = numpy.zeros(self.imSize)
        critDist     = 0 # int(self.shapeSize/6)
        padDist      = int(self.shapeSize/6)
        shapeMatrix  = numpy.array(shapeMatrix)

        if len(shapeMatrix.shape) < 2:
            shapeMatrix = numpy.expand_dims(shapeMatrix, axis=0)

        if shapeMatrix.size == 0:  # this means we want only a vernier
            patch = numpy.zeros((self.shapeSize, self.shapeSize))
        else:
            patch = numpy.zeros((shapeMatrix.shape[0]*self.shapeSize + (shapeMatrix.shape[0]-1)*critDist + 1,
                                 shapeMatrix.shape[1]*self.shapeSize + (shapeMatrix.shape[1]-1)*critDist + 1))

            for row in range(shapeMatrix.shape[0]):
                for col in range(shapeMatrix.shape[1]):

                    firstRow = row*(self.shapeSize + critDist)
                    firstCol = col*(self.shapeSize + critDist)

                    patch[firstRow:firstRow+self.shapeSize, firstCol:firstCol+self.shapeSize] = self.drawShape(shapeMatrix[row,col], offset, offset_size)

        if vernier_in:

            firstRow = int((patch.shape[0]-self.shapeSize)/2) # + 1  # small adjustments may be needed depending on precise image size
            firstCol = int((patch.shape[1]-self.shapeSize)/2) # + 1
            patch[firstRow:(firstRow+self.shapeSize), firstCol:firstCol+self.shapeSize] += self.drawVernier(offset, offset_size)
            patch[patch > 1.0] = 1.0

        if fixed_position is None:
            firstRow = random.randint(padDist, self.imSize[0] - (patch.shape[0]+padDist))
            firstCol = random.randint(padDist, self.imSize[1] - (patch.shape[1]+padDist))
        else:
            n_elements = [max(shapeMatrix.shape[0],1), max(shapeMatrix.shape[1],1)]  # because vernier alone has matrix [[]] but 1 element
            firstRow = fixed_position[0]-int(self.shapeSize*(n_elements[0]-1)/2)  # this is to always have the vernier at the fixed_position
            firstCol = fixed_position[1]-int(self.shapeSize*(n_elements[1]-1)/2)  # this is to always have the vernier at the fixed_position

        image[firstRow:firstRow+patch.shape[0], firstCol:firstCol+patch.shape[1]] = patch

        min_distance = 0

        if vernier_ext:
            ver_size = self.shapeSize
            ver_patch = numpy.zeros((ver_size, ver_size)) + self.drawVernier(offset, offset_size)
            x = firstRow
            y = firstCol

            flag = 0
            while x+ver_size + min_distance >= firstRow and x <= min_distance + firstRow + patch.shape[0] and y+ ver_size >=firstCol and y<=firstCol + patch.shape[1]:

                x = numpy.random.randint(padDist, self.imSize[0] - (ver_size+padDist))
                y = numpy.random.randint(padDist, self.imSize[1] - (ver_size+padDist))

                flag += 1
                if flag > 15:
                    print("problem in finding space for the extra vernier")

            image[x: x + ver_size, y: y + ver_size] = ver_patch

        if noise_patch is not None:
            image[noise_patch[0]:noise_patch[0] + self.shapeSize // 2,
            noise_patch[1]:noise_patch[1] + self.shapeSize // 2] = self.drawNoise()

        return image


    def plotStim(self, vernier, shapeMatrix):

        plt.figure()
        plt.imshow(self.drawStim(vernier, shapeMatrix))
        plt.show()


    def show_Batch(self, batchSize, ratios, noiseLevel=0.0, normalize=False, fixed_position=None, shapeMatrix=[], noise_patch=None, offset=None):
        # input a configuration to display
        batchImages, batchLabels = self.generate_Batch(batchSize, ratios, noiseLevel=noiseLevel, normalize=normalize, fixed_position=fixed_position, shapeMatrix=shapeMatrix, noise_patch=noise_patch, offset=offset)

        for n in range(batchSize):
            plt.figure()
            plt.imshow(batchImages[n, :, :, 0])
            plt.title('Label, mean, stdev = ' + str(batchLabels[n]) + ', ' + str(
                numpy.mean(batchImages[n, :, :, 0])) + ', ' + str(numpy.std(batchImages[n, :, :, 0])))
            plt.show()


    def generate_Batch(self, batchSize, ratios, noiseLevel=0.0, normalize=False, fixed_position=None, shapeMatrix=None, noise_patch=None, offset=None, offset_size=None, fixed_noise=None):

        # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier ext; 3-vernier inside random shape; 4- vernier inside shapeMatrix
        # in case ratio didn't fit required size, standard output
        if len(ratios)!= 4:
            ratios = [1., 1., 1., 0.]

        #  Normalize ratios by batchSize, then manage rounding errors with while
        ratios = [int(float(i)*batchSize / sum(ratios)) for i in ratios]

        while sum(ratios) < batchSize:
            ratios[0] += 1

        # Define attributes of all 3 groups (could be dictionnary)
        v_map = ((True, False), (False, False), (True, False),(False, True))
        shape_map = ([], shapeMatrix, None, shapeMatrix)

        # Define output
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        vernierLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        # generate images, master loop
        n_precedent=0
        offset_ori = offset
        for grp in range(4):
            N = ratios[grp]
            for n in range(N):
                n_true = n_precedent + n
                if offset_ori is None:
                    offset = random.randint(0, 1)
                img = self.drawStim(vernier_ext=v_map[grp][0], shapeMatrix=shape_map[grp], vernier_in=v_map[grp][1], fixed_position=fixed_position, offset=offset, offset_size=offset_size, noise_patch=noise_patch)
                if normalize:
                    img = (img - numpy.mean(img)) / numpy.std(img)

                batchImages[n_true, :, :] = img
                vernierLabels[n_true] = -offset + 1


            n_precedent += N

        # Make it suitable for alexnet: RGB and noise added
        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow
        batchImages = numpy.tile(batchImages, (1, 1, 1, 3))
        if fixed_noise is not None:
            batchImages += fixed_noise
        else:
            batchImages += numpy.random.normal(0, noiseLevel, size=(batchImages.shape))

        return batchImages, vernierLabels


    def generate_Batch_uncrowding(self, batchSize, noiseLevel=0.0, normalize=False, fixed_position=None):

        # generates a batch of uncrowding conditions with different shapes

        # Define output
        batchImages = numpy.ndarray(shape=(batchSize, self.imSize[0], self.imSize[1]), dtype=numpy.float32)
        vernierLabels = numpy.zeros(batchSize, dtype=numpy.float32)

        # generate images, master loop
        n_precedent=0
        for n in range(batchSize):
            shapeType = random.randint(1, 6)
            shapeMatrix = [shapeType] * 7
            offset = random.randint(0, 1)
            img = self.drawStim(vernier_ext=False, shapeMatrix=shapeMatrix, vernier_in=True, fixed_position=fixed_position, offset=offset)
            if normalize:
                img = (img - numpy.mean(img)) / numpy.std(img)
            batchImages[n, :, :] = img
            vernierLabels[n] = -offset + 1

        # Make it suitable for alexnet: RGB and noise added
        batchImages = numpy.expand_dims(batchImages, -1)  # need to a a fourth dimension for tensorflow
        batchImages = numpy.tile(batchImages, (1, 1, 1, 3))
        batchImages += numpy.random.normal(0, noiseLevel, size=(batchImages.shape))

        return batchImages, vernierLabels


if __name__ == "__main__":

    imgSize = (227, 227)
    shapeSize = 18
    barWidth = 1
    rufus = StimMaker(imgSize, shapeSize, barWidth)
    ratios = [0,0,1,0]  # ratios : 0 - vernier alone; 1- shapes alone; 2- Vernier outside shape; 3-vernier inside shape
    batchSize = 1
    t = all_test_shapes()
    for matrix in t:
        rufus.show_Batch(batchSize,ratios, noiseLevel=0.1, normalize=False, fixed_position=None, shapeMatrix=matrix)