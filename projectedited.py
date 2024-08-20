import numpy as np
import cv2

from sklearn.metrics import pairwise
from sklearn.metrics import pairwise_distances

background=None

accumulated_weight=0.5

#setting the region of interests.

roi_top=20
roi_bottom=300
roi_left=600
roi_right=300

#finding the average background value 

def calc_avg(frame,accumalted_weight):
    global background
    if background is None:
        background=frame.copy().astype('float')

        return None
    cv2.accumulateWeighted(frame,background,accumalted_weight)


#tresholding the hand segment from the ROI 

def segment(frame,threshold_min=25):
    diff=cv2.absdiff(background.astype('uint8'),frame)

    ret,thresholded=cv2.threshold(diff,threshold_min,255,cv2.THRESH_BINARY)
    contours,hierarchy = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if len(contours)==0:
        return None
    else:
        #hand as the largest contour 
        hand_segment=max(contours,key= cv2.contourArea)

        return(thresholded,hand_segment)
    
#counting the fingers using convex hull 

#convex hull basically draws a polygon around the external points of the frame 
#calculating the extreme points

def count_fingers(thresholded, hand_segment):
    convex = cv2.convexHull(hand_segment)

    # Check if convex is a single point (scalar)
    if len(convex) == 1:
        print("Convex hull is a single point. Finger counting not possible.")
        return 0  # Or handle the case as needed

    convex = convex.reshape(-1, 2)  # Reshape to 2D array with 2 columns

    min_idx = np.argmin(convex[:, 1])
    max_idx = np.argmax(convex[:, 1])
    min_x_idx = np.argmin(convex[:, 0])
    max_x_idx = np.argmax(convex[:, 0])

    top = tuple(convex[min_idx])
    bottom = tuple(convex[max_idx])
    left = tuple(convex[min_x_idx])
    right = tuple(convex[max_x_idx])
    xc=(left[0]+right[0])//2
    yc=(top[1]+bottom[1])//2

    distance=pairwise.euclidean_distances([[xc, yc]], convex)[0]
    max_distance=np.max(distance)
    radius=int(0.9*max_distance)
    circumference=(2*np.pi*radius)

    circular_roi=np.zeros(thresholded.shape, dtype="uint8")
    cv2.circle(circular_roi,(xc,yc),radius,255,10)
    

    circular_roi=cv2.bitwise_and(thresholded,thresholded,mask=circular_roi)
    
    contours, _ = cv2.findContours(circular_roi.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    count=0

    for cnt in contours:
        (x,y,w,h)=cv2.boundingRect(cnt)

        out_of_wrist=(yc+(yc*0.25))>(y+h)

        limit_points=((circumference*0.25)>cnt.shape[0])

        if out_of_wrist and limit_points:
            count+=1

    return count 

cam=cv2.VideoCapture(0)

no_frames=0
while True:
    ret,frame=cam.read()
    frame_copy=frame.copy()

    roi=frame[roi_top:roi_bottom,roi_right:roi_left]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(7,7),0)

    if no_frames<60:
        calc_avg(gray,accumulated_weight)

        if no_frames<=59:
            cv2.putText(frame_copy," obtaining background",(200,300),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0))
            cv2.imshow("fingercount",frame_copy)

    else:

        hand=segment(gray)
        if hand is not None:
            thresholded,hand_segment=hand
#draw contours across the hand 
            cv2.drawContours(frame_copy,[hand_segment+(roi_right,roi_top)],-1,(0,0,0),5)

            fingers=count_fingers(thresholded,hand_segment)
            cv2.putText(frame_copy,str(fingers),(70,50),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            cv2.imshow("threshold image",thresholded)

    cv2.rectangle(frame_copy,(roi_left,roi_top),(roi_right,roi_bottom),(0,0,255),5)

    no_frames+=1
    cv2.imshow("fingercounts",frame_copy)

    if cv2.waitKey(1) & 0xFF ==27:
        break

cam.release()
cv2.destroyAllWindows()