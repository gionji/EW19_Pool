import cv2
import numpy as np
import math
import imutils

title_window = "topooo"

### bal colors

ball_white  = [233, 212, 198, 255, 255, 255,  255, 255, 255]
ball_red    = [173,   0,   0, 255, 121, 255,   255, 0, 0]
ball_blue   = [  0,   0, 148, 138, 125, 255,    0, 0, 255]
ball_yellow = [221, 175,   0, 255, 241,  90,   255, 255, 0]
ball_orange = [180,  90,   0, 255, 174, 112,   255, 100, 0]
ball_violet = [ 39,   0,  69,  97,  84, 155,   255, 50, 255]
ball_purple = [ 34,   0,  93, 165,  77, 127,   255, 0, 255]
ball_green  = [  0,  89,  59,  41, 142,  96,    0, 255, 0]
ball_black  = []

table =  [47, 86, 97, 89, 195, 140]

ballColor = []

balls = [ball_white , ball_red, ball_blue, ball_yellow, ball_orange, ball_violet, ball_purple, ball_green]

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
		
		## I just resized the image to a quarter of its original size
		#image = cv2.resize(image, (0, 0), None, .25, .25)

		#trasform to HSV
		rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

		final = rgb
		for ball in balls:
			mask = cv2.inRange(rgb, (ball[0], ball[1], ball[2]), (ball[3], ball[4], ball[5]))
			mask = cv2.erode(mask, None, iterations=1)
			mask = cv2.dilate(mask, None, iterations=3)
			#mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
			center = None

			if len(cnts) > 0:
				c = max(cnts, key=cv2.contourArea)
				((x, y), radius) = cv2.minEnclosingCircle(c)
				
				circle_color = (ball[8], ball[7], ball[6])
				if radius > 10 and radius < 40:
					cv2.circle(resized, (int(x), int(y)), int(radius), circle_color, 2)


			# combine original and mask image
		#horizontal_concat = np.concatenate((resized, mask), axis=1)
		
		if(ret):
			cv2.imshow('frame',resized)
			cv2.imshow('mask',mask)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()



if __name__ == "__main__":
	main()
