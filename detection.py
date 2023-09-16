import cv2 as cv
import mediapipe as mp
import time
import utils, math
import numpy as np
import sqlite3
import os
import datetime
import pygame

pygame.mixer.init()
soundObj = pygame.mixer.Sound('resource/ringtone.mp3')

# variables 
frame_counter =0
video_name = ""
DISTANCE_COUNTER =0
TOTAL_WRONG = 0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX

# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

# iris indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
KNOWN_INTEROCULAR_DISTANCE = 6.5  # In centimeters
ASSUMED_FOCAL_LENGTH = 1000       # In pixels (adjust as needed)

map_face_mesh = mp.solutions.face_mesh

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# database connection
def findNearestSession():
    with open("data/SessionCount.txt", 'r') as f:
        line = f.read()
        return line

def changeNearestSession():
    with open("data/SessionCount.txt", 'r') as f:
        line = int(f.read())
    with open("data/SessionCount.txt", 'w') as f:
        if line == 4:
            f.write("1")
        else:
            f.write(str(line+1))
database_name = "data/SessionDb.db"
conn = sqlite3.connect(database_name)
cur = conn.cursor()
current_session = ""
def database_work():
    global current_video_dir
    global current_session
    current_video_dir = 'data/Session' + findNearestSession()
    current_session = "SessionTrack" + findNearestSession()
    current_table_drop = "DROP TABLE IF EXISTS " + current_session
    current_table_creat = "CREATE TABLE " + current_session + "(Id INTEGER PRIMARY KEY AUTOINCREMENT, Time TEXT, VideoPath TEXT)"
    for f in os.listdir(current_video_dir):
        os.remove(os.path.join(current_video_dir, f))

    cur.execute(current_table_drop)
    cur.execute(current_table_creat)
    
database_work()
conn.commit()
# camera object 
camera = cv.VideoCapture(0)
# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def findDistance(x1, y1, x2, y2):
    dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist

# Function to calculate distance from interocular distance
def calculate_distance(interocular_distance):
    distance = (KNOWN_INTEROCULAR_DISTANCE * ASSUMED_FOCAL_LENGTH) / interocular_distance
    return distance

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
    
    if cropped_right.shape[0] == 0 or cropped_right.shape[1] == 0 or cropped_left.shape[0] == 0 or cropped_left.shape[1] == 0:
        return None, None

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Starting time here
start_time = time.time()

# Variable to store the previous interocular distance
previous_interocular_distance = None

endtimer_text=""
timer_text=""
distance_text=""
ln_text="" 
rn_text="" 
incli_text=""
fps_text=""
distance_to_camera = 0
ret, frame = camera.read()
review = 0
wrong = 0

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def reviewMode(review_link):
    global camera
    global review
    global TOTAL_WRONG
    TOTAL_WRONG = 0
    review = 1
    camera = cv.VideoCapture(review_link)

def exitReviewMode(temp):
    global camera
    global review
    global TOTAL_WRONG
    TOTAL_WRONG = temp
    review = 0
    camera = cv.VideoCapture(0)

def detectionLoop():
    global frame, ret
    global timer_text, endtimer_text
    global total_wrong_text
    global distance_text, ln_text, rn_text, incli_text
    global fps_text
    global frame_counter 
    global video_name 
    global DISTANCE_COUNTER 
    global TOTAL_WRONG
    global writer
    global timer, wrong, distance_to_camera

    with map_face_mesh.FaceMesh(
    max_num_faces = 1, 
    refine_landmarks=True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5
) as face_mesh:
        frame_counter = frame_counter + 1 # frame counter
        ret, frame = camera.read() # getting frame from camera 
        if not ret: 
            return 0# no more frames break
        #  resizing frame

        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width= frame.shape[:2]
        rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        results  = face_mesh.process(rgb_frame)
        keypoints = pose.process(rgb_frame)
        if keypoints is not None and keypoints.pose_landmarks is not None:
                lm = keypoints.pose_landmarks
            # Rest of your code
        else:
            # Handle case where no landmarks are detected
                return  # Skip this iteration and move to the next frame

        img_h, img_w = frame.shape[:2]
        
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            
            mesh_points=np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            (l_cx, l_cy),l_radius = cv.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy),r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
           
            #get shoulder and nose
            
            lmPose = mp_pose.PoseLandmark

            # Acquire the landmark coordinates.
            # Once aligned properly, left or right should not be a concern.      
            # Left shoulder.
            l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * frame_width)
            l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * frame_height)
            # Right shoulder
            r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * frame_width)
            r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * frame_height)
            
            nose_x = int(lm.landmark[lmPose.NOSE].x * frame_width)
            nose_y = int(lm.landmark[lmPose.NOSE].y * frame_height)

            # Measure the interocular distance
            left_eye_center = np.array(mesh_coords[LEFT_EYE[8]])
            right_eye_center = np.array(mesh_coords[RIGHT_EYE[8]])
            interocular_distance = np.linalg.norm(left_eye_center - right_eye_center)

            # Calculate the distance to the camera
            distance_to_camera = calculate_distance(interocular_distance)
            distance_text = f'{round(distance_to_camera, 2)} cm'
            
            distance_ln = findDistance(l_shldr_x, l_shldr_y, nose_x, nose_y)
            distance_rn = findDistance(r_shldr_x, r_shldr_y, nose_x, nose_y)
            ln_text = f'Left shoulder - nose: {round(distance_ln,2)}'
            rn_text = f'Right shoulder - nose: {round(distance_rn,2)}'

            if (distance_ln - distance_rn > 60):
                incli_text = "Right bending"
            elif  (distance_rn - distance_ln > 60):
                incli_text = "Left bending"
            else:
                incli_text = "Center"

            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv.LINE_AA )
            cv.circle(frame, (l_shldr_x, l_shldr_y), 7, yellow, -1)
            cv.circle(frame, (r_shldr_x, r_shldr_y), 7, pink, -1)
            cv.circle(frame, (nose_x, nose_y), 7, yellow, -1)
            

            if distance_to_camera < 45 or distance_to_camera>80 or (abs(distance_ln - distance_rn)) > 60:
                if (abs(distance_ln - distance_rn)) >60:
                    color = red
                else:
                    color = green
                
                if distance_to_camera < 45 or distance_to_camera>80:
                    colorE = utils.RED
                else:
                    colorE = utils.GREEN

                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, colorE, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, colorE, 1, cv.LINE_AA)
                cv.line(frame, (l_shldr_x, l_shldr_y), (nose_x, nose_y), color, 4)
                cv.line(frame, (r_shldr_x, r_shldr_y), (nose_x, nose_y), color, 4)
                #start timer + video start
                DISTANCE_COUNTER+=1
                if DISTANCE_COUNTER == 1:
                    timer = time.time()
                    if review == 0:
                        video_name = current_video_dir + "/Distance_Track" + "_" + str(TOTAL_WRONG + 1) + ".mp4"
                        writer= cv.VideoWriter(video_name, cv.VideoWriter_fourcc(*'DIVX'), 20, (frame_width,frame_height))
                endtimer = int(time.time()-timer)
                endtimer_text = f'Irregular: {int(endtimer//60)} minutes, {int(endtimer%60)} seconds'
                if (endtimer < 2):
                    wrong = 0
                if (endtimer >= 2):
                    
                    if endtimer == 2:
                        wrong +=1
                    if wrong == 1: #write to database
                        TOTAL_WRONG +=1
                        if review == 0:
                            soundObj.play()
                            sqlsyn = "INSERT INTO " + current_session + "(Time, VideoPath) VALUES ('" + str(datetime.datetime.now()) +"', '" + video_name +"')"
                            cur.execute(sqlsyn)
                            conn.commit()

                if review == 0:
                    
                    img2 = frame.copy()
                    done_text = distance_text + '    ' + incli_text
                    cv.putText(img2, done_text, (10, frame_height - 80), FONTS, 0.9, yellow, 2)
                    cv.putText(img2, rn_text, (10, frame_height - 50), FONTS, 0.9, yellow, 2)
                    cv.putText(img2, ln_text, (10, frame_height - 20), FONTS, 0.9, yellow, 2)
                    writer.write(img2)

            else:
                soundObj.stop()
                DISTANCE_COUNTER = 0
                wrong =0
                cv.line(frame, (l_shldr_x, l_shldr_y), (nose_x, nose_y), green, 4)
                cv.line(frame, (r_shldr_x, r_shldr_y), (nose_x, nose_y), green, 4)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
            total_wrong_text = f'Total Wrong Distance: {TOTAL_WRONG}'
            
            

        # Timer
        end_time = time.time()-start_time
        end_minutes = int(end_time // 60)
        end_seconds = int(end_time % 60)
        
        # Calculating  frame per seconds FPS
        fps = frame_counter/end_time
        
        timer_text = f'Timer: {end_minutes} minutes, {end_seconds} seconds'
        fps_text = f'FPS: {round(fps,1)}' #optional?
        
        #cv.imshow('Test', frame)