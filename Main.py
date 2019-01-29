import cv2
import numpy as np
import math
import imutils

title_window = "topooo"

RADIUS_MIN  = 16
RADIUS_MAX  = 25
RADIUS_DRAW = 20

### bal colors
ball_white  = [227, 216, 181, 255, 252, 240,  255, 255, 255]
ball_red    = [192,   0,   0, 255, 154, 100,  255,   0,   0]
ball_blue   = [  0,   0,  94, 138,  83, 255,    0,   0, 255]
ball_yellow = [221, 175,   0, 255, 241,  90,  255, 255,   0]
ball_orange = [180,  90,   0, 255, 174, 112,  255, 100,   0]
ball_violet = [ 39,   0,  69,  97,  84, 155,  255,  50, 255]
ball_purple = [ 34,   0,  93, 165,  77, 127,  255,   0, 255]
ball_green  = [  0,  57,   0,  34, 124,  70,    0, 255,   0]
ball_black  = []
ball_brown  = [100,   0,  0,  133, 106,  85,  255, 120,  90]
ball_pink   = [255, 109, 106, 255, 193, 195,  255, 128, 255]

table =  [47, 86, 97, 89, 195, 140]

ballColor = []

balls = [ball_white, ball_brown, ball_pink,  ball_red, ball_blue, ball_yellow, ball_green]

table_vert = [ [520, 46], [ 27 , 56] ,[ 57, 325] ,[508, 308] ]

def cv_size(img):
    return tuple(img.shape[1::-1])

def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
	return rect

def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
 
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
 
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
 
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
	# return the warped image
	return warped


def main():

	global coordinates

	cap = cv2.VideoCapture(0)

	while(cap.isOpened()):

		# capture frame
		ret, frame = cap.read()
		#resize
		resized = imutils.resize(frame, width=600)
		size = cv_size(resized)
		plot = np.zeros((size[1], size[0], 3))

		#### DETECT TABLE ####################################################################################
		# convert the resized image to grayscale, blur it slightly, and threshold it
		gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (7, 7), 0)
		thresh  = cv2.threshold(blurred, 200, 250, cv2.THRESH_BINARY_INV)[1]

		# find contours in the thresholded image and initialize the shape detector
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnt = max(cnts, key=cv2.contourArea)

		#for cnt in cnts:
		rect = cv2.minAreaRect(cnt)
		table_external = cv2.boxPoints(rect)
		table_external = np.int0(table_external)

		### two approach for hull
		# 1)			
		epsilon = 0.01 * cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, epsilon, True)
		# 2)
		hull = cv2.convexHull(cnt)

		cv2.drawContours(plot,[hull],0,(0,0,255),2)
		#####################################################################################################


		#trasform to HSV
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		for ball in balls:
			mask = cv2.inRange(rgb, (ball[0], ball[1], ball[2]), (ball[3], ball[4], ball[5]))
			mask = cv2.erode( mask, None, iterations=1)
			mask = cv2.dilate(mask, None, iterations=3)
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
			center = None

			mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

			for c in cnts:
				#c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				
				circle_color = (ball[8], ball[7], ball[6])
				if radius > RADIUS_MIN and radius < RADIUS_MAX:
					cv2.circle(plot, (int(x), int(y)), int(RADIUS_DRAW), circle_color, 2)

		
		#res = cv2.bitwise_and(rgb,rgb, mask=mask)
		#res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)


		table_vert = approx[0][0], approx[1][0], approx[2][0], approx[3][0]

		if approx.shape[0] == 4:
			pts = np.array(table_vert , dtype = "float32")
			stretch = four_point_transform(resized, pts)
			plot    = four_point_transform(plot,   pts)
			cv2.imshow('frame',stretch)

		plot_resized = imutils.resize(plot, width=600)

		if(ret):
			cv2.imshow('frame',resized)
			cv2.imshow('mask', plot_resized)
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
