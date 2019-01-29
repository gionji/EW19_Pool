import cv2
import numpy as np
import math
import imutils

title_window = "topooo"
controls_title_window = "controls"
alpha_slider_max = 255

LOW_RED = 0
LOW_GREEN = 1
LOW_BLUE = 2
HIGH_RED = 3
HIGH_GREEN = 4
HIGH_BLUE = 5

ball = [173,   0,   0, 255, 121, 255]

radius_min = 10
radius_max = 40

#balls = [ball_white , ball_red, ball_blue, ball_yellow, ball_orange, ball_violet, ball_purple, ball_green]

### sliders callbacks
def on_trackbar_lR(val):
	global ball
	ball[LOW_RED] = int(val) 

def on_trackbar_lG(val):
	global ball
	ball[LOW_GREEN] = int(val) 

def on_trackbar_lB(val):
	global ball
	ball[LOW_BLUE] = int(val) 


def on_trackbar_hR(val):
	global ball
	ball[HIGH_RED] = int(val) 

def on_trackbar_hG(val):
	global ball
	ball[HIGH_GREEN] = int(val) 

def on_trackbar_hB(val):
	global ball
	ball[HIGH_BLUE] = int(val) 

def on_trackbar_radmin(val):
	global radius_min
	radius_min = int(val) 

def on_trackbar_radmax(val):
	global radius_max
	radius_max = int(val) 



def cv_size(img):
	return tuple(img.shape[1::-1])




def main():

	#### controls window ####
	cv2.namedWindow(controls_title_window)
	
	cv2.createTrackbar("trackbar_lR", controls_title_window , ball[LOW_RED], alpha_slider_max, on_trackbar_lR)
	cv2.createTrackbar("trackbar_lG", controls_title_window , ball[LOW_GREEN], alpha_slider_max, on_trackbar_lG)
	cv2.createTrackbar("trackbar_lB", controls_title_window , ball[LOW_BLUE], alpha_slider_max, on_trackbar_lB)

	cv2.createTrackbar("trackbar_hR", controls_title_window , ball[HIGH_RED], alpha_slider_max, on_trackbar_hR)
	cv2.createTrackbar("trackbar_hG", controls_title_window , ball[HIGH_GREEN], alpha_slider_max, on_trackbar_hG)
	cv2.createTrackbar("trackbar_hB", controls_title_window , ball[HIGH_BLUE], alpha_slider_max, on_trackbar_hB)

	cv2.createTrackbar("trackbar_rad_min", controls_title_window , radius_min, 80, on_trackbar_radmin)
	cv2.createTrackbar("trackbar_rad_max", controls_title_window ,radius_max, 80, on_trackbar_radmax)

	global coordinates

	#### acquire image
	cap = cv2.VideoCapture(0)

	while(cap.isOpened()):

		#### capture frame
		ret, frame = cap.read()
		#### resize
		resized = imutils.resize(frame, width=600)
		
		#### I just resized the image to a quarter of its original size
		#image = cv2.resize(image, (0, 0), None, .25, .25)

		#### BGR --> RGB
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
		hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

		final = rgb

		#### create mask
		mask = cv2.inRange(rgb, (ball[0], ball[1], ball[2]), (ball[3], ball[4], ball[5]))
		# edit mask
		mask = cv2.erode(mask, None, iterations=1)
		mask = cv2.dilate(mask, None, iterations=2)

		res = cv2.bitwise_and(rgb,rgb, mask= mask)
		res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
		
		#### find contours
		cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
		center = None

		if len(cnts) > 0:
			c = max(cnts, key=cv2.contourArea)
			((x, y), radius) = cv2.minEnclosingCircle(c)
				
			circle_color = (255, 255, 255)
			if radius > radius_min and radius < radius_max:
				cv2.circle(res, (int(x), int(y)), int(radius), circle_color, 2)


		#### combine original and mask image
		#horizontal_concat = np.concatenate((resized, mask), axis=1)
		
		if(ret):
			cv2.imshow('frame', resized)
			cv2.imshow('mask', res)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
