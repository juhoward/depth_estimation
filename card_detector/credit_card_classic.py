'''
Uses classical methods to detect the presence of a rectangle
with the same aspect ratio as a credit card.

Step 1: convert to grayscale
Step 2: blur image to reduce noise
Step 3: edge  detection (Canny used)
Step 4: find contours
Step 5: detector shapes close to rectangular
Step 6: filter proposed rectangles by area
Step 7: filter proposed rectangles by aspect ratio

bounding boxes are returned and should be filtered once more to find those
near the face.
'''
from .shapedetector import ShapeDetector
from .validator import Validator
import imutils
import cv2
import numpy as np


class CardDetector(object):
    def __init__(self) -> None:
        # credit card dimensions (h, w)
        self.ratio = 53.98 / 85.6

    def get_threshold_vals(self, img, sigma = .33):
        # helper that automatically finds threshold for Canny function
        med = np.median(img)
        lower = int(max(0, (1-sigma) * med))
        upper = int(min(255, (1 - sigma) * med))
        return lower, upper

    def process_image(self, image):
        # load the image and resize it to a smaller factor so that
        # the shapes can be approximated better
        # resized = imutils.resize(image, width=300)
        # ratio = image.shape[0] / float(resized.shape[0])
        # convert the resized image to grayscale, blur it slightly,
        # and threshold it
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lower, upper = self.get_threshold_vals(img, sigma=.003)
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        # thresh = cv2.threshold(blurred, lower, 255, cv2.THRESH_BINARY)[1]
        
        edges = cv2.Canny(blurred, lower, upper, apertureSize=3)
        # # find contours in the thresholded image and initialize the
        # # # shape detector
        cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        sd = ShapeDetector()
        # holder for boxes that meet basic criteria
        proposals = []
        # loop over the contours
        for c in cnts:
            M = cv2.moments(c)
            # compute the center of the contour, then detect the name of the
            # shape using only the contour
            if M["m00"] > 300:
                cX = int((M["m10"] / M["m00"]))# * ratio)
                cY = int((M["m01"] / M["m00"]))# * ratio)

                shape = sd.detect(c)
                if shape == "card":
                    # fit box to countour
                    box = cv2.minAreaRect(c)
                    # fit to height / width ratio of cards
                    aspect_ratio = box[1][1] / box[1][0]
                    if abs(aspect_ratio - self.ratio) < .05:
                        box = cv2.boxPoints(box)
                        box = np.int0(box)
                        proposals.append(box)
                        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
                        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
        # show the output image
        # cv2.imshow("Image", image)
        # cv2.imshow("Image", edges)
        return proposals