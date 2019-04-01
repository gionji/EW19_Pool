
import numpy as np
from collections import deque
import cv2
import imutils
from imutils.video import VideoStream
import time
import math

#######################
#    TABLE SIZE FORCED AT LINE
#######################
SM_SIZE = 8  # Size of the smoothin window to stabilize the FG Mask
MIN_CIRCLE_RADIUS = 10  # Minimum radius for a blob to be considered a ball
MAX_CIRCLE_RADIUS = 25  # Minimum radius for a blob to be considered a ball
DRAW_RADIUS = 20  # Size of the drawn circle
BORDER_THICK = 30  # Table-Field border size
font = cv2.FONT_HERSHEY_SIMPLEX  # Frame/table text font
FRAME_FONT_SCALE = 1
TABLE_FONT_SCALE = 1
TABLE_DRAW_COLOR = (0, 255, 0)
FIELD_DRAW_COLOR = (255, 0, 0)

# HSV color ranges (min, max) and BGR value for circles drawing
color_dict = {0 :{'black' : [(0, 0, 0),       (121, 255, 63),  (0 ,0 ,0)]},
              1 :{'white' : [(0, 0, 113),   (42, 104, 247),   (255 ,255 ,255)]},
              2 :{'red'   : [(0, 153, 183), (3, 248, 255),   (0 ,0 ,255)]},
              3 :{'yellow': [(16, 134, 234),   (24, 255, 255),  (0 ,255 ,255)]},
              4 :{'orange': [(6, 134, 234),   (13, 255, 255),  (0 ,140 ,255)]},
              5 :{'maroon': [(134, 167, 83),   (208, 255, 168), (0 ,0 ,128)]},
              6 :{'blue'  : [(99, 151, 60),    (125, 255, 255), (255 ,0 ,0)]},
              7 :{'purple': [(119, 82, 31),   (156, 255, 130), (128 ,0 ,128)]},
              8 :{'green' : [(66, 137, 65),    (95, 255, 104),  (0 ,255 ,0)]}
              }

table_color =  [(45, 53, 77),    (75, 210, 183),  (0 ,255 ,0)]
shadow_color = [(46, 50, 72),    (80, 153, 101),  (0 ,255 ,0)]


MAX_SIZE = 1500
BALLS_NUM = 8
IMAGE_WIDTH = 800

RADIUS_MIN  = 6
RADIUS_MAX  = 50
RADIUS_DRAW = 10


def detectTable(frame):
    table_external = None
    # blur frame
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    # convert to hsv color space
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    Lower, Upper = table_color[0], table_color[1]
    mask = cv2.inRange(hsv, Lower, Upper)

    mask = cv2.dilate(mask, None, iterations=5)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        cnt = max(cnts, key=cv2.contourArea)

        # for cnt in cnts:
        rect = cv2.minAreaRect(cnt)
        table_external = cv2.boxPoints(rect)
        table_external = order_points(table_external)
        table_external = np.int0(table_external)

        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx  = cv2.approxPolyDP(cnt, epsilon, True)

    return table_external, mask



def stretchToRect(frame, approx):
    if len(approx) == 4:
        table_vert = approx[0], approx[1], approx[2], approx[3]
        pts = np.array(table_vert , dtype = "float32")
        stretch = four_point_transform(frame, pts)
        return stretch
    else:
        return None


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


def getField(tableRect):

    # table parameters
    # M = cv2.moments(tableRect)
    # x1, y1, w, h = cv2.boundingRect(tableRect)
    # cy = int(M['m01']/M['m00'])
    # cx = int(M['m10']/M['m00'])

    field_rect = np.array([
                  ( tableRect[0][0] + BORDER_THICK, tableRect[0][1] + BORDER_THICK ) ,
                  ( tableRect[1][0] + BORDER_THICK, tableRect[1][1] - BORDER_THICK ) ,
                  ( tableRect[2][0] - BORDER_THICK, tableRect[2][1] - BORDER_THICK ) ,
                  ( tableRect[3][0] - BORDER_THICK, tableRect[3][1] + BORDER_THICK )] )

    return field_rect


def getBallsFromTable(frame, field_mask):

    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    lower, upper = table_color[0], table_color[1]
    mask = cv2.inRange(hsv, lower, upper)

    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=1)

    mask = 255 - mask

    mouse = cv2.bitwise_and(mask, mask, mask=field_mask)

    cnts = cv2.findContours(mouse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

    return cnts


def remove_holes(raw_mask, off_set=30):
    ym = int(raw_mask.shape[0]/2)
    xm = int(raw_mask.shape[1]/2)
    om = off_set # int(off_set/2)
    raw_mask[-off_set:, -off_set:] = 0
    raw_mask[-off_set:, :off_set] = 0
    raw_mask[:off_set, -off_set:] = 0
    raw_mask[:off_set:, :off_set] = 0
    raw_mask[:off_set, xm-om: xm+om] = 0
    raw_mask[-off_set:, xm-om: xm+om] = 0
    return raw_mask


def clean_mask(foremask, erode_iterations=2, dilate_iteraions=7):
    foremask = cv2.erode(foremask, None, iterations=erode_iterations)
    foremask = cv2.dilate(foremask, None, iterations=dilate_iteraions)
    return foremask

# initiate video capture for video file
cap = cv2.VideoCapture(0)

print (int(cap.get(cv2.CAP_PROP_FPS)))

# def main():

fgbg = cv2.createBackgroundSubtractorMOG2()

fms_couter = 0
sm_mask = None

centers = list()
last_frame_centers = list()

table_found = False
field_found = False

while cap.isOpened():

    # time.sleep(.05)

    # read a frame
    ret, frame = cap.read()

    if ret:
        # resize frame
        frame = imutils.resize(frame, width=IMAGE_WIDTH)
        frame = cv2.medianBlur(frame, 7)
        
        # Table border from green
        if not table_found:
            table_rect, table_mask = detectTable(frame)
            if table_rect is not None:
                table_found = True
                xlt = min(table_rect[:, 1])
                xrt = max(table_rect[:, 1])
                yut = min(table_rect[:, 0])
                ylt = max(table_rect[:, 0])
                ################################################################################################################
                #
                #                            FORCING TABLE BOUNDARIES
                #
                xlt, xrt, yut, ylt = (50, 490, 20,760)
                table_rect = [[yut, xlt], [yut, xrt], [ylt, xrt], [ylt, xlt]]
                ################################################################################################################
                print("Detected Table at positions: ", xlt, xrt, yut, ylt)
                showing_frame = frame.copy()
                cv2.rectangle(showing_frame, (yut, xlt), (ylt, xrt), TABLE_DRAW_COLOR, 3)
                cv2.putText(showing_frame, 'Table', (yut, xlt), font, TABLE_FONT_SCALE, TABLE_DRAW_COLOR, 2, cv2.LINE_AA)
                cv2.imshow('Table', frame)

        # print rect on the table
        if table_rect is None:
            continue

        # Find Playning Rectangle inside table
        if not field_found:
            field_rect = getField(table_rect)
            if field_rect is not None:
                field_found = True
                xl = min(field_rect[:, 1])
                xr = max(field_rect[:, 1])
                yu = min(field_rect[:, 0])
                yl = max(field_rect[:, 0])
                print("Detected Field at positions: ", xl, xr, yu, yl)
                cv2.rectangle(showing_frame, (yu, xl), (yl, xr),(255, 0, 0),3)
                cv2.putText(showing_frame, 'Field', (yu, xl), font, FRAME_FONT_SCALE, FIELD_DRAW_COLOR, 2, cv2.LINE_AA)
                cv2.imshow('Table', showing_frame)
        
        if field_rect is None:
            continue
        
        field_image = frame[xl: xr, yu: yl]
        cropped_size = field_image.shape
        #cv2.imshow("Field detection", field_image)
        
        # ALL'INIZIO CALCOLO IL BG
        if fms_couter < 60:
            fg_mask = fgbg.apply(field_image)

            fms_couter += 1
            continue

        # POI CERCO LE PALLE
        subtracted = np.zeros((cropped_size[0], cropped_size[1], 3), dtype=np.uint8)
        s2 = subtracted.copy()
        s2t = subtracted.copy()
        fg_mask = fgbg.apply(field_image, fg_mask, 0)

        
        fg_mask_c = clean_mask(fg_mask, erode_iterations=3, dilate_iteraions=7)
        fg_mask_c = cv2.medianBlur(fg_mask_c, 7)
        fg_mask_c = remove_holes(fg_mask_c)
        
        
        if sm_mask is None:
            sm_mask = [fg_mask_c] * SM_SIZE
        else:
            sm_mask.pop(0)
            sm_mask.append(fg_mask_c)

        sfg = np.median(np.array(sm_mask), axis=0)
        

        subtracted[fg_mask_c > 128] = field_image[fg_mask_c > 128]
        s2[sfg > 128] = field_image[sfg > 128]
        s2 = clean_mask(s2, erode_iterations=1, dilate_iteraions=4)
        

        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        edges = cv2.Canny(frame, 100, 200)

        cv2.imshow("subtracted", subtracted)
        cv2.imshow("subtracted2", s2)
        cv2.imshow("laplacian", laplacian)
        cv2.imshow("edges", edges)

        blurred = cv2.medianBlur(s2, 7)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)


        #cv2.imshow("fgmask", fg_mask)
        cv2.imshow("subtracted gray", gray)

        
        
        # Detect blobs
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                                   param1=10, param2=20, minRadius=MIN_CIRCLE_RADIUS, maxRadius=MAX_CIRCLE_RADIUS)

        blob_areas = list()
        
        print(circles)

        if circles is not None:
            if len(circles) > 0:

                cc = circles[0]
                for i in range(len(cc)):                
                        cv2.circle(field_image, (cc[i, 0], cc[i, 1]), DRAW_RADIUS, (0,255,255), 2)
                        cv2.putText(field_image, str(cc[i, 0])+ " " + str(cc[i, 1]) , (cc[i, 0], cc[i, 1]), font, 0.4, TABLE_DRAW_COLOR, 2, cv2.LINE_AA)
                               

        cv2.imshow('Field image', field_image)
        # cv2.imshow('gray', cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

        # if the 'q' key is pressed, stop the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()
