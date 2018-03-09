import math

import cv2
import numpy as np
import scipy
from scipy import ndimage, spatial, misc
import progressbar

import transformations

import random

# GLOBAL VARIABLE
NUMBER_OF_SCALES = 4
sigma_pyramid = 1

ALL_VALUES_ALL_SCALES = False
USE_VARIANCE = False


def inbounds(shape, indices):
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


# Keypoint detectors ##########################################################
class KeypointDetector(object):
    def detectKeypoints(self, image):
        raise NotImplementedError()


class DummyKeypointDetector(KeypointDetector):

    def detectKeypoints(self, image):
        image = image.astype(np.float32)
        image /= 255.
        features = []
        height, width = image.shape[:2]

        for y in range(height):
            for x in range(width):
                r = image[y, x, 0]
                g = image[y, x, 1]
                b = image[y, x, 2]

                if int(255 * (r + g + b) + 0.5) % 100 == 1:
                    # If the pixel satisfies this meaningless criterion,
                    # make it a feature.

                    f = cv2.KeyPoint()
                    f.pt = (x, y)
                    # Dummy size
                    f.size = 10
                    f.angle = 0
                    f.response = 10

                    features.append(f)

        return features


class HarrisKeypointDetector(KeypointDetector):
    def saveHarrisImage(self, harrisImage, srcImage):

        outshape = [harrisImage.shape[0], harrisImage.shape[1], 3]
        outImage = np.zeros(outshape)
        # Make a grayscale srcImage as a background
        srcNorm = srcImage * (0.3 * 255 / (np.max(srcImage) + 1e-50))
        outImage[:, :, :] = np.expand_dims(srcNorm, 2)

        # Add in the harris keypoints as red
        outImage[:, :, 2] += harrisImage * (4 * 255 / (np.max(harrisImage)) + 1e-50)
        cv2.imwrite("harris.png", outImage)

    # Compute harris values of an image.
    def computeHarrisValues(self, srcImage):
        '''
		Input:
			srcImage -- Grayscale input image in a numpy array with
						values in [0, 1]. The dimensions are (rows, cols).
		Output:
			harrisImage -- numpy array containing the Harris score at
						   each pixel.
			orientationImage -- numpy array containing the orientation of the
								gradient at each pixel in degrees.
		'''
        height, width = srcImage.shape[:2]

        harrisImage = np.zeros(srcImage.shape[:2])
        orientationImage = np.zeros(srcImage.shape[:2])

        # TODO 1: Compute the harris corner strength for 'srcImage' at
        # each pixel and store in 'harrisImage'.  See the project page
        # for direction on how to do this. Also compute an orientation
        # for each pixel and store it in 'orientationImage.'
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        Ix = ndimage.sobel(srcImage, mode='reflect', axis=1)
        Iy = ndimage.sobel(srcImage, mode='reflect', axis=0)

        A = ndimage.gaussian_filter(np.square(Ix), sigma=0.5)
        B = ndimage.gaussian_filter(np.multiply(Ix, Iy), sigma=0.5)
        C = ndimage.gaussian_filter(np.square(Iy), sigma=0.5)

        # det(H) = AC-B^2	trace(H) = A+C
        harrisImage = np.multiply(A, C) - np.square(B) - 0.1 * np.square(A + C)
        orientationImage = np.degrees(np.arctan2(Iy, Ix))

        print(np.max(harrisImage))
        # TODO-BLOCK-END

        # Save the harris image as harris.png for the website assignment
        self.saveHarrisImage(harrisImage, srcImage)

        return harrisImage, orientationImage

    def computeLocalMaxima(self, harrisImage):
        '''
		Input:
			harrisImage -- numpy array containing the Harris score at
						   each pixel.
		Output:
			destImage -- numpy array containing True/False at
						 each pixel, depending on whether
						 the pixel value is the local maxima in
						 its 7x7 neighborhood.
		'''
        destImage = np.zeros_like(harrisImage, np.bool)

        # TODO 2: Compute the local maxima image
        # TODO-BLOCK-BEGIN
        destImage = np.equal(ndimage.filters.maximum_filter(harrisImage, size=7), harrisImage)

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return destImage

    def detectKeypoints(self, image):

        image = image.astype(np.float32)
        image /= 255.
        height, width = image.shape[:2]
        features = []

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        harrisImage, orientationImage = self.computeHarrisValues(grayImage)
        harrisMaxImage = self.computeLocalMaxima(harrisImage)

        for y in range(height):
            for x in range(width):
                if not harrisMaxImage[y, x]:
                    continue

                f = cv2.KeyPoint()
                f.size = 10
                f.pt = (x, y)
                f.angle = orientationImage[y, x]
                f.response = harrisImage[y, x]

                features.append(f)

        if ALL_VALUES_ALL_SCALES:
            return features * NUMBER_OF_SCALES
        else:
            return features


class ORBKeypointDetector(KeypointDetector):
    def detectKeypoints(self, image):
        '''
		Input:
			image -- uint8 BGR image with values between [0, 255]
		Output:
			list of detected keypoints, fill the cv2.KeyPoint objects with the
			coordinates of the detected keypoints, the angle of the gradient
			(in degrees) and set the size to 10.
		'''
        detector = cv2.ORB()
        return detector.detect(image)


## Feature descriptors #########################################################


class FeatureDescriptor(object):
    # Implement in child classes
    def describeFeatures(self, image, keypoints):
        '''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
			descriptors at the specified coordinates
		Output:
			Descriptor numpy array, dimensions:
				keypoint number x feature descriptor dimension
		'''
        raise NotImplementedError


class SimpleFeatureDescriptor(FeatureDescriptor):
    # TODO: Implement parts of this function
    def describeFeatures(self, image, keypoints):
        '''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
						 descriptors at the specified coordinates
		Output:
			desc -- K x 25 numpy array, where K is the number of keypoints
		'''
        image = image.astype(np.float32)
        image /= 255.
        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        desc = np.zeros((len(keypoints), 5 * 5))

        for i, f in enumerate(keypoints):
            x, y = f.pt
            x, y = int(x), int(y)

            # TODO 4: The simple descriptor is a 5x5 window of intensities
            # sampled centered on the feature point. Store the descriptor
            # as a row-major vector. Treat pixels outside the image as zero.
            # TODO-BLOCK-BEGIN
            s1, s2 = grayImage.shape
            if x in [0, 1, s2 - 1, s2 - 2] or y in [0, 1, s1 - 1, s1 - 2]:
                grayImage_pad = np.pad(grayImage, 2, 'constant', constant_values=0)
                descriptor = grayImage_pad[y:y + 5, x:x + 5]
            else:
                x_beg, y_beg = x - 2, y - 2
                x_end, y_end = x + 3, y + 3
                descriptor = grayImage[y_beg:y_end, x_beg:x_end]
            desc[i, :] = descriptor.flatten()

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return desc


class MOPSFeatureDescriptor(FeatureDescriptor):

    def describeFeatures(self, image, keypoints):

        if ALL_VALUES_ALL_SCALES:
            # Reduce the number of keypoints to relevant values.
            keypoints = keypoints[:len(keypoints) / NUMBER_OF_SCALES]

        final_size = len(keypoints)

        image = image.astype(np.float32)
        image /= 255.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        windowSize = 8

        scaled_blurred_image = {0: ndimage.gaussian_filter(gray_image, 0.5)}
        for scale in range(1, NUMBER_OF_SCALES):
            gray_image = ndimage.gaussian_filter(gray_image, sigma_pyramid)

            gray_image = misc.imresize(gray_image, 1/2.)
            width, height = gray_image.shape

            # Skip images that have become too small.
            if width < 40 or height < 40:
                break

            scaled_blurred_image[scale] = ndimage.gaussian_filter(gray_image, 0.5)
            cv2.imwrite("harris_" + str(scale) + ".png", scaled_blurred_image[scale])

        if ALL_VALUES_ALL_SCALES:
            desc = np.zeros((len(keypoints) * NUMBER_OF_SCALES, windowSize * windowSize))
        else:
            desc = np.zeros((len(keypoints), windowSize * windowSize))

        accepted_key_points = set()
        with progressbar.ProgressBar(max_value=len(keypoints)) as progress:
            status = 0
            for i, f in enumerate(keypoints):

                x, y = f.pt
                angle = f.angle

                c = np.cos(np.radians(-angle))
                s = np.sin(np.radians(-angle))

                # USE_RANDOM
                # descriptor_cross_scale = []

                # USE_VARIANCE
                instance_std = 1e-5
                descriptor = np.zeros((1, 64))

                for scale, single_image in scaled_blurred_image.items():

                    if scale != 0:
                        x, y = int(x/2), int(y/2)

                    if not ALL_VALUES_ALL_SCALES:
                        # If a particular scale, x, y has already been recorded, don't add again, if using variance
                        if (scale, x, y) in accepted_key_points:
                            continue

                    T1 = np.array([[1, 0, -x], [0, 1, -y], [0, 0, 1]])
                    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    S = np.array([[0.2, 0, 0], [0, 0.2, 0], [0, 0, 1]])
                    T2 = np.array([[1, 0, 4], [0, 1, 4], [0, 0, 1]])

                    transMx = T2.dot(S.dot(R.dot(T1)))
                    transMx = transMx[:2, :]

                    destImage = cv2.warpAffine(single_image, transMx, (windowSize, windowSize), flags=cv2.INTER_LINEAR)

                    # think of a good metric to choose, if at all.
                    # if USE_VARIANCE:
                    #     scale_std = destImage.std()
                    #     if scale_std > instance_std:
                    #         desc[i, :] = np.divide(destImage - destImage.mean(), scale_std).flatten()
                    #         instance_std = scale_std

                    std = destImage.std()
                    if std <= 1e-5:
                        descriptor = np.zeros((1, 64))
                    else:
                        descriptor = np.divide(destImage - destImage.mean(), std)

                    desc[i + scale * final_size, :] = descriptor.flatten()

                    # RANDOM SCALE
                    # descriptor_cross_scale.append((scale, descriptor.flatten()))

                status += 1
                progress.update(status)

                # RANDOM SCALE
                # randomly_chosen_descriptor = descriptor_cross_scale[random.randint(0, len(descriptor_cross_scale) - 1)]
                # desc[i, :] = randomly_chosen_descriptor[1]
                # accepted_key_points.add((randomly_chosen_descriptor[0], x, y))

        return desc


class ORBFeatureDescriptor(KeypointDetector):
    def describeFeatures(self, image, keypoints):
        '''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
			descriptors at the specified coordinates
		Output:
			Descriptor numpy array, dimensions:
				keypoint number x feature descriptor dimension
		'''
        descriptor = cv2.ORB()
        kps, desc = descriptor.compute(image, keypoints)
        if desc is None:
            desc = np.zeros((0, 128))

        return desc


# Compute Custom descriptors (extra credit)
class CustomFeatureDescriptor(FeatureDescriptor):
    def describeFeatures(self, image, keypoints):
        '''
		Input:
			image -- BGR image with values between [0, 255]
			keypoints -- the detected features, we have to compute the feature
			descriptors at the specified coordinates
		Output:
			Descriptor numpy array, dimensions:
				keypoint number x feature descriptor dimension
		'''
        raise NotImplementedError('NOT IMPLEMENTED')


## Feature matchers ############################################################


class FeatureMatcher(object):
    def matchFeatures(self, desc1, desc2):
        '''
		Input:
			desc1 -- the feature descriptors of image 1 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
			desc2 -- the feature descriptors of image 2 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
		Output:
			features matches: a list of cv2.DMatch objects
				How to set attributes:
					queryIdx: The index of the feature in the first image
					trainIdx: The index of the feature in the second image
					distance: The distance between the two features
		'''
        raise NotImplementedError

    # Evaluate a match using a ground truth homography.  This computes the
    # average SSD distance between the matched feature points and
    # the actual transformed positions.
    @staticmethod
    def evaluateMatch(features1, features2, matches, h):
        d = 0
        n = 0

        for m in matches:
            id1 = m.queryIdx
            id2 = m.trainIdx
            ptOld = np.array(features2[id2].pt)
            ptNew = FeatureMatcher.applyHomography(features1[id1].pt, h)

            # Euclidean distance
            d += np.linalg.norm(ptNew - ptOld)
            n += 1

        return d / n if n != 0 else 0

    # Transform point by homography.
    @staticmethod
    def applyHomography(pt, h):
        x, y = pt
        d = h[6] * x + h[7] * y + h[8]

        return np.array([(h[0] * x + h[1] * y + h[2]) / d,
                         (h[3] * x + h[4] * y + h[5]) / d])


class SSDFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
		Input:
			desc1 -- the feature descriptors of image 1 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
			desc2 -- the feature descriptors of image 2 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
		Output:
			features matches: a list of cv2.DMatch objects
				How to set attributes:
					queryIdx: The index of the feature in the first image
					trainIdx: The index of the feature in the second image
					distance: The distance between the two features
		'''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return matches

        # TODO 7: Perform simple feature matching.  This uses the SSD

        for index, feature_1 in enumerate(desc1):
            distance = desc2 - feature_1
            ssd = np.array([np.sum(np.square(single_dist)) for single_dist in distance])
            minimum_two = ssd.argsort()[:2]

            matches.append(cv2.DMatch(index, minimum_two[0], ssd[minimum_two[0]]))

        # TODO-BLOCK-BEGIN

        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class RatioFeatureMatcher(FeatureMatcher):
    def matchFeatures(self, desc1, desc2):
        '''
		Input:
			desc1 -- the feature descriptors of image 1 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
			desc2 -- the feature descriptors of image 2 stored in a numpy array,
				dimensions: rows (number of key points) x
				columns (dimension of the feature descriptor)
		Output:
			features matches: a list of cv2.DMatch objects
				How to set attributes:
					queryIdx: The index of the feature in the first image
					trainIdx: The index of the feature in the second image
					distance: The ratio test score
		'''
        matches = []
        # feature count = n
        assert desc1.ndim == 2
        # feature count = m
        assert desc2.ndim == 2
        # the two features should have the type
        assert desc1.shape[1] == desc2.shape[1]

        if desc1.shape[0] == 0 or desc2.shape[0] == 0:
            return []

        # TODO 8: Perform ratio feature matching.
        for index, feature_1 in enumerate(desc1):
            distance = desc2 - feature_1
            ssd = np.array([np.sum(np.square(single_dist)) for single_dist in distance])
            minimum_two = ssd.argsort()[:2]

            matches.append(cv2.DMatch(index, minimum_two[0], ssd[minimum_two[0]] / ssd[minimum_two[1]]))

        # This uses the ratio of the SSD distance of the two best matches
        # and matches a feature in the first image with the closest feature in the
        # second image.
        # Note: multiple features from the first image may match the same
        # feature in the second image.
        # You don't need to threshold matches in this function
        # TODO-BLOCK-BEGIN
        # raise Exception("TODO in features.py not implemented")
        # TODO-BLOCK-END

        return matches


class ORBFeatureMatcher(FeatureMatcher):
    def __init__(self):
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        super(ORBFeatureMatcher, self).__init__()

    def matchFeatures(self, desc1, desc2):
        return self.bf.match(desc1.astype(np.uint8), desc2.astype(np.uint8))
