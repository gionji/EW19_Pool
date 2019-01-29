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
ball_blue   = [  0,   0, 94,  138,  83, 255,    0,   0, 255]
ball_yellow = [221, 175,   0, 255, 241,  90,  255, 255,   0]
ball_orange = [180,  90,   0, 255, 174, 112,  255, 100,   0]
ball_violet = [ 39,   0,  69,  97,  84, 155,  255,  50, 255]
ball_purple = [ 34,   0,  93, 165,  77, 127,  255,   0, 255]
ball_green  = [  0,  57,  0,  34, 124,  70,   0, 255,   0]
ball_black  = []
ball_brown  = [100,  0,  0, 133, 106,  85, 255, 120,  90]
ball_pink   = [255, 109, 106, 255, 193, 195, 255, 128, 255]

table =  [47, 86, 97, 89, 195, 140]

ballColor = []

balls = [ball_brown, ball_pink,  ball_red, ball_blue, ball_yellow,  ball_green]

def cv_size(img):
    return tuple(img.shape[1::-1])


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

		#### DETECT TABLE
		# convert the resized image to grayscale, blur it slightly, and threshold it
		gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(gray, (5, 5), 0)
		thresh  = cv2.threshold(blurred, 200, 250, cv2.THRESH_BINARY_INV)[1]

		# find contours in the thresholded image and initialize the shape detector
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		cnt = cnts[0]		

		rect = cv2.minAreaRect(cnt)
		table_external = cv2.boxPoints(rect)
		table_external = np.int0(table_external)
		cv2.drawContours(plot,[table_external],0,(0,0,255),2)

		print(table_external)		


		#trasform to HSV
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		final = rgb
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
		
		if(ret):
			cv2.imshow('frame',resized)
			cv2.imshow('mask', plot)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
