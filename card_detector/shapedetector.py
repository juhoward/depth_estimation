''' Rectangle Detector modified to find ID cards '''

import cv2

vid = '/home/jhoward/facedepth/occlusion_1_10.mp4'


class ShapeDetector:
	def __init__(self):
		pass
	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = "unidentified"
		# contour perimeter
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = "triangle"
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		# elif len(approx) == 4:
		# 	# compute the bounding box of the contour and use the
		# 	# bounding box to compute the aspect ratio
		# 	(x, y, w, h) = cv2.boundingRect(approx)
		# 	ar = w / float(h)
		# 	# a square will have an aspect ratio that is approximately
		# 	# equal to one, otherwise, the shape is a rectangle
		# 	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

		elif len(approx) > 3 and len(approx) < 9: #9
			shape = "card"
		# otherwise, we assume the shape is a circle
		else:
			shape = "circle"
		# return the name of the shape
		return shape