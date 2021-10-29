import numpy as np
from skimage import morphology
import math
import cv2
import mrc
from skimage.morphology import skeletonize_3d
from fil_finder import FilFinder2D
import astropy.units as u
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from skimage.transform import probabilistic_hough_line

"""
20210317 update
use polyDp approximate fit skeleton with straight line
20210318 update
use end to end method
20210322
read start file and write coordinates to assigned folder 
"""

def generateLineTem(box,length,width):
    empty = np.zeros([box,box])
    line = np.ones([length,width])
    empty[int(box/2-length/2):int(box/2+length/2),int(box/2-width/2):int(box/2+width/2)] = line
    return 1-empty

def rotate_image(image, angle, BackGroudValue = 1):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=BackGroudValue)
  return result

class FilamentPicker:

    def __init__(self,image_path,pixel):
        self.pixel = pixel
        self.image_path = image_path
        self.MaxBoundary = 255
        self.denose_p1 = 8
        self.denose_p2 = 42
        self.rescal_ratio = 0.1
        self.blur_size = 3

        self.LineLength = 40
        self.LineWidth = 4

        self.filament = None

        self.cutoff = 80 # Important parameters
        self.minimal_length = 50

        self.epsilon = 0.008
        #self.line_thr = 10
        #self.line_len = 20
        #self.line_gap = 5

    def picking(self):

        #Read image and preprocess
        image = mrc.imread(self.image_path)
        image = np.array(image[0, :, :])
        image = self.MaxBoundary * (image - image.min()) / (image.max() - image.min())
        image = image.astype(np.uint8)

        #Denose and shrink image to 10%, for acceleration

        denoise = cv2.fastNlMeansDenoising(image, self.denose_p1, self.denose_p2)
        half = cv2.resize(denoise, (0, 0), fx=self.rescal_ratio, fy=self.rescal_ratio)
        new = cv2.blur(half, (self.blur_size, self.blur_size))

        #Find ccoeff map, using solid cylinder template

        l = (generateLineTem(self.LineLength, self.LineLength, self.LineWidth) * self.MaxBoundary).astype(np.uint8)
        empty = None
        method = cv2.TM_CCOEFF
        for i in range(0, 360, 10):
            unit = rotate_image(l, i, self.MaxBoundary)
            if empty is None:
                d1, d2 = cv2.matchTemplate(new, unit, cv2.TM_CCOEFF).shape
                empty = np.zeros((d1, d2, 1))
            empty = np.concatenate((empty, cv2.matchTemplate(new, unit, method).reshape(d1, d2, 1)), 2)
        #plt.imshow(empty.max(2), 'gray')
        #Process the map, cut off at 85 percentile
        a = empty.max(2)
        a = self.MaxBoundary  * (a - a.min()) / (a.max() - a.min())
        a = a.astype(np.uint8)

        end_method = True

        if end_method:
            cutoff = self.cutoff
            pos = a > np.percentile(a.reshape(-1), cutoff)
            neg = a <= np.percentile(a.reshape(-1), cutoff)
            a[neg] = 0
            a[pos] = 1
        else:
            s = np.bitwise_and(a >= np.percentile(a, 60), a < np.percentile(a, 95))
            s_r = np.bitwise_not(s)
            a[s] = 1
            a[s_r] = 0

        self.trace = a

        #generate contours and select large contours
        contours, hierarchy = cv2.findContours(a, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        area = [cv2.contourArea(i) for i in contours]

        mask = a.copy()
        mask[:] = 0
        for i, item in enumerate(area):
            if item > 50:
                mask = cv2.drawContours(mask, [contours[i]], -1, (255, 255, 255), -1)
        skeleton = skeletonize_3d(mask)

        #find longest axis
        fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40 * u.pix, skel_thresh=10 * u.pix, prune_criteria='length')

        #mark coordinate in origin image

        skeleton = fil.skeleton_longpath.copy()
        valuable_brach = (fil.skeleton - fil.skeleton_longpath).copy()
        #back = np.zeros(half.shape)
        gy = round((half.shape[0] - skeleton.shape[0]) / 2)
        gx = round((half.shape[1] - skeleton.shape[1]) / 2)
        #back[gy:gy + skeleton.shape[0], gx:gx + skeleton.shape[1]] = fil.skeleton_longpath
        #draw = half.copy()
        #draw[back != 0] = draw.max()
        #plt.imshow(draw, 'gray')

        estimation = []

        skeleton[skeleton != 0] = 255
        skeleton = skeleton.astype(np.uint8)
        contours, hie = cv2.findContours(skeleton, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #apply polyDP approx here

        for idx, contour in enumerate(contours):
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if cv2.arcLength(approx, False) > 75:
                estimation.append(approx)

        #add valuable braches
        valuable_brach[valuable_brach != 0] = 255
        valuable_brach = valuable_brach.astype(np.uint8)
        contours_branch, hie = cv2.findContours(valuable_brach, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for idx, contour in enumerate(contours_branch):
            epsilon = self.epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, closed=True)
            if cv2.arcLength(approx, False) > 75:
                estimation.append(approx)


        # generate ends point
        full_skeleton = fil.skeleton.copy()
        #ends = probabilistic_hough_line(full_skeleton, threshold=self.line_thr, line_length=self.line_len, line_gap=self.line_gap)
        gy = round((half.shape[0] - full_skeleton.shape[0]) / 2)
        gx = round((half.shape[1] - full_skeleton.shape[1]) / 2)
        filament = []
        rescal = 10
        minimal_length = self.minimal_length/rescal

        for contour in estimation:
            for ci, point in enumerate(contour):
                if ci==0:
                    continue
                #print(point)
                (y1, x1) = contour[ci - 1][0]
                (y2, x2) = point[0]
                y1 = (y1 + gy) * rescal
                x1 = (x1 + gx) * rescal
                y2 = (y2 + gy) * rescal
                x2 = (x2 + gx) * rescal
                length = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                if length >= minimal_length:
                    filament.append([(y1, x1), (y2, x2)])
        self.filament = filament

class RelionStarCoordinate:
    #output coordinate file with relion star format
    def __init__(self,filament,output_path='None.star',pixel_size=1.06):
        self.filament=filament
        self.pixel_size=pixel_size
        self.output_path=output_path

        self.description=['# RELION; version 3.0-beta-2, helical autopicking NTU NISB WB','','data_','',
                          'loop_ ','_rlnCoordinateX #1 ','_rlnCoordinateY #2 ',
                          '_rlnClassNumber #3 ','_rlnAnglePsi #4', '_rlnAutopickFigureOfMerit #5 ']
        self.end='\n \n'
        self.output=''
        self.FielnameOut=output_path[:-4]+'_manualpick.star'


    def StarGenerator(self):
        coortext=[]
        for line in self.filament:
            (x1, y1), (x2, y2) = line
            end_point_1 = '{CoorX:12f}{CoorY:13f}{Cnumer:13n}{Angpsi:13f}{merit:13f} '.format(
                CoorX=x1, CoorY=y1, Cnumer=-999, Angpsi=-999, merit=-999)
            end_point_2 = '{CoorX:12f}{CoorY:13f}{Cnumer:13n}{Angpsi:13f}{merit:13f} '.format(
                CoorX=x2, CoorY=y2, Cnumer=-999, Angpsi=-999, merit=-999)
            coortext.append(end_point_1)
            coortext.append(end_point_2)

        output=self.description+coortext
        output='\n'.join(output)
        self.output=output+self.end

        f=open(self.FielnameOut,'w')
        f.write(self.output)
        f.close()

import sys
from collections import OrderedDict
import copy
import numpy as np


LABELS = {
    'rlnVoltage': float,
    'rlnDefocusU': float,
    'rlnDefocusV': float,
    'rlnDefocusAngle': float,
    'rlnSphericalAberration': float,
    'rlnDetectorPixelSize': float,
    'rlnCtfFigureOfMerit': float,
    'rlnMagnification': float,
    'rlnAmplitudeContrast': float,
    'rlnImageName': str,
    'rlnOriginalName': str,
    'rlnCtfImage': str,
    'rlnCoordinateX': float,
    'rlnCoordinateY': float,
    'rlnCoordinateZ': float,
    'rlnNormCorrection': float,
    'rlnMicrographName': str,
    'rlnGroupName': str,
    'rlnGroupNumber': str,
    'rlnOriginX': float,
    'rlnOriginY': float,
    'rlnAngleRot': float,
    'rlnAngleTilt': float,
    'rlnAnglePsi': float,
    'rlnClassNumber': int,
    'rlnLogLikeliContribution': float,
    'rlnRandomSubset': int,
    'rlnParticleName': str,
    'rlnOriginalParticleName': str,
    'rlnNrOfSignificantSamples': float,
    'rlnNrOfFrames': int,
    'rlnMaxValueProbDistribution': float
}


class Label():
    def __init__(self, labelName):
        self.name = labelName
        # Get the type from the LABELS dict, assume str by default
        self.type = LABELS.get(labelName, str)

    def __str__(self):
        return self.name

    def __cmp__(self, other):
        return self.name == str(other)


class Item():
    """
    General class to store data from a row. (e.g. Particle, Micrograph, etc)
    """

    def copyValues(self, other, *labels):
        """
        Copy the values form other object.
        """
        for l in labels:
            setattr(self, l, getattr(other, l))

    def clone(self):
        return copy.deepcopy(self)


class MetaData():
    """ Class to parse Relion star files
    """
    def __init__(self, input_star=None,tube=1000000):
        self.tube = tube
        if input_star:
            self.read(input_star)
        else:
            self.clear()

    def clear(self):
        self._labels = OrderedDict()
        self._data = []

    def _setItemValue(self, item, label, value):
        setattr(item, label.name, label.type(value))

    def _addLabel(self, labelName):
        self._labels[labelName] = Label(labelName)

    def read(self, input_star):
        self.clear()
        found_label = False
        f = open(input_star)
        start = True
        tube = self.tube

        for line in f:
            values = line.strip().split()

            if not values: # empty lines
                continue

            if values[0].startswith('_rln'):  # Label line
                # Skip leading underscore in label name
                self._addLabel(labelName=values[0][1:])
                found_label = True

            elif found_label:  # Read data lines after at least one label
                if start:
                    self.addLabels(['rlnHelicalTubeID','rlnAngleTiltPrior','rlnAnglePsiPrior','rlnHelicalTrackLength','rlnAnglePsiFlipRatio'])
                    start = False
                #values.append(str(int(tube%9998)+1))
                values.append(str(tube))
                values.append('0')
                values.append('0')
                values.append('100')
                values.append('0')
                tube = tube + 1
                # Iterate in pairs (zipping) over labels and values in the row
                item = Item()
                # Dynamically set values, using label type (str by default)
                for label, value in zip(self._labels.values(), values):
                    self._setItemValue(item, label, value)

                self._data.append(item)

        f.close()

    def _write(self, output_file):
        output_file.write("\ndata_\n\nloop_\n")
        line_format = ""

        # Write labels and prepare the line format for rows
        for i, l in enumerate(self._labels.values()):
            output_file.write("_%s #%d \n" % (l.name, i+1))
            # Retrieve the type of the label
            t = l.type
            if t is float:
                line_format += "%%(%s)f \t" % l.name
            elif t is int:
                line_format += "%%(%s)d \t" % l.name
            else:
                line_format += "%%(%s)s \t" % l.name

        line_format += '\n'

        for item in self._data:
            output_file.write(line_format % item.__dict__)

        output_file.write('\n')

    def write(self, output_star):
        output_file = open(output_star, 'w')
        self._write(output_file)
        output_file.close()

    def printStar(self):
        self._write(sys.stdout)

    def size(self):
        return len(self._data)

    def __len__(self):
        return self.size()

    def __iter__(self):
        for item in self._data:
            yield item

    def getLabels(self):
        return [l.name for l in self._labels.values()]

    def setLabels(self, **kwargs):
        """ Add (or set) labels with a given value. """
        for key, value in kwargs.iteritems():
            if key not in self._labels:
                self._addLabel(labelName=key)

        for item in self._data:
            for key, value in kwargs.iteritems():
                self._setItemValue(item, self._labels[key], value)

    def _iterLabels(self, labels):
        """ Just a small trick to accept normal lists or *args
        """
        for l1 in labels:
            if isinstance(l1, list):
                for l2 in l1:
                    yield l2
            else:
                yield l1

    def addLabels(self, *labels):
        """
        Register labes in the metadata, but not add the values to the rows
        """
        for l in self._iterLabels(labels):
            if l not in self._labels.keys():
                self._addLabel(l)

    def removeLabels(self, *labels):
        for l in self._iterLabels(labels):
            if l in self._labels:
                del self._labels[l]

    def addItem(self, item):
        """ Add a new item to the MetaData. """
        self._data.append(item)

    def setData(self, data):
        """ Set internal data with new items. """
        self._data = data

    def addData(self, data):
        """ Add new items to internal data. """
        for item in data:
            self.addItem(item)



star_file = '/home/xuch0013/20210521_sp100/full_data/relion/CtfFind/job003/micrographs_ctf.star'
output_folder = '/home/xuch0013/20210521_sp100/full_data/relion/ManualPick/job006/movies'
pixel_size = 1.1
minimal_length = (14*50) / pixel_size

star = MetaData(star_file)

for i in tqdm(range(len(star._data))):
    fullpath = './' + star._data[i].rlnMicrographName
    pick = FilamentPicker(fullpath,pixel_size)
    pick.minimal_length = minimal_length
    try:
        pick.picking()
        out_path = output_folder + '/' + star._data[i].rlnMicrographName.split('/')[-1]
        writer = RelionStarCoordinate(pick.filament,out_path,pixel_size)
        writer.StarGenerator()
    except:
        continue



#dir_path = '/media/ntu/disk4/20210112_FilamentAutopickingSample/UMOD/5Images'
#starpath = 'path'


"""file_list_raw = os.listdir(dir_path)
file_list = []
for file in file_list_raw:
    if file[-4:] == '.mrc':
        file_list.append(file)

for file in tqdm(file_list):
    fullpath = dir_path+'/'+file
    pick = FilamentPicker(fullpath,pixel_size)
    pick.minimal_length = minimal_length
    pick.picking()
    writer = RelionStarCoordinate(pick.filament,pick.image_path,pixel_size)
    writer.StarGenerator()"""
