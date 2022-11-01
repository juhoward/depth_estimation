from shapedetector import ShapeDetector
import imutils
import cv2
import numpy as np


class CardDetector(object):
    def __init__(self) -> None:
        self.ratio = 53.98 / 85.6
    def get_threshold_vals(self, img, sigma = .33):
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
                    # multiply the contour (x, y)-coordinates by the resize ratio,
                    # then draw the contours and the name of the shape on the image
                    # c = c.astype("float")
                    # c *= ratio
                    # c = c.astype("int")
                    box = cv2.minAreaRect(c)
                    # width * height
                    # area = box[1][0] * box[1][1]
                    # fit to height / width ratio of cards
                    aspect_ratio = box[1][1] / box[1][0]
                    if abs(aspect_ratio - self.ratio) < .05:
                        box = cv2.boxPoints(box)
                        box = np.int0(box)
                        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
                        cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
        # show the output image
        # cv2.imshow("Image", image)
        # cv2.imshow("Image", edges)
        return box