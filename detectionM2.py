import cv2
import mediapipe as mp
import time
import math as m
import numpy as np
import sqlite3
import os
import datetime
import pygame

pygame.mixer.init()
soundObj = pygame.mixer.Sound('resource/ringtone.mp3')

# database connection
def findNearestSession():
    with open("data/SessionCountM2.txt", 'r') as f:
        line = f.read()
        return line

def changeNearestSession():
    with open("data/SessionCountM2.txt", 'r') as f:
        line = int(f.read())
    with open("data/SessionCountM2.txt", 'w') as f:
        if line == 4:
            f.write("1")
        else:
            f.write(str(line+1))
database_name = "data/SessionDbM2.db"
conn = sqlite3.connect(database_name)
cur = conn.cursor()
current_session = ""

def database_work():
    global current_video_dir
    global current_session
    current_video_dir = 'data/Session' + findNearestSession() +"M2"
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
droidcam_ip = "172.20.21.235"
droidcam_port = "4747"
droidcam_url = f"http://{droidcam_ip}:{droidcam_port}/video"
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(droidcam_url)
# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree

# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
video_name = ""
WRONG_COUNTER =0
TOTAL_WRONG = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#

# Starting time here
start_time = time.time()

# Variable to store the previous interocular distance
previous_interocular_distance = None

endtimer_text=""
timer_text=""
fps_text=""

review = 0
success, image = cap.read()



def reviewMode(review_link):
    global cap
    global review
    global TOTAL
    TOTAL_WRONG = 0
    review = 1
    cap = cv2.VideoCapture(review_link)

def exitReviewMode(temp):
    global cap
    global review
    global TOTAL_WRONG
    TOTAL_WRONG = temp
    review = 0
    #cap = cv2.VideoCapture(droidcam_url)
    cap = cv2.VideoCapture(0)


def detectionLoop():
    global success, image
    global timer_text, endtimer_text
    global total_wrong_text
    global video_name 
    global WRONG_COUNTER 
    global TOTAL_WRONG
    global writer
    global timer, wrong, distance_to_camera
    global neck_inclination_text, torso_inclination_text
    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)

    while cap.isOpened():
    # Capture frames.
        success, image = cap.read()
        if not success:
            return 0
        # Get fps.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # Get height and width.
        h, w = image.shape[:2]

        # Convert the BGR image to RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image.
        keypoints = pose.process(image)

        # Convert the image back to BGR.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    

        # Use lm and lmPose as representative of the following methods.
        # lm = keypoints.pose_landmarks
        if keypoints is not None and keypoints.pose_landmarks is not None:
            lm = keypoints.pose_landmarks
            # Rest of your code
        else:
            # Handle case where no landmarks are detected
            continue  # Skip this iteration and move to the next frame
        lmPose = mp_pose.PoseLandmark

        # Acquire the landmark coordinates.
        # Once aligned properly, left or right should not be a concern.      
        # Left shoulder.
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        # Right shoulder
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)
        # Left ear.
        l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
        l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)
        # Left hip.
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

        # Calculate distance between left shoulder and right shoulder points.

        # Assist to align the camera to point at the side view of the person.

        # Calculate angles.
        neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

        

        # Put text, Posture and angle inclination.
        # Text string for display.
        torso_inclination_text = 'Torso: ' + str(int(torso_inclination)) + 'degrees'
        neck_inclination_text = 'Neck: ' + str(int(neck_inclination)) + 'degrees'

        # Determine whether good posture or bad posture.
        # The threshold angles have been set based on intuition.
        # Draw landmarks.
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (l_ear_x, l_ear_y), 7, yellow, -1)

        # Let's take y - coordinate of P3 100px above x1,  for display elegance.
        # Although we are taking y = 0 while calculating angle between P1,P2,P3.
        cv2.circle(image, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Similarly, here we are taking y - coordinate 100px above x1. Note that
        # you can take any value for y, not necessarily 100 or 200 pixels.
        cv2.circle(image, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

        if neck_inclination >= 40 or torso_inclination > 10:
                WRONG_COUNTER+=1
                if WRONG_COUNTER == 1:
                    timer = time.time()
                    if review == 0:
                        video_name = current_video_dir + "/Pose_Track" + "_" + str(TOTAL_WRONG + 1) + ".mp4"
                        writer= cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

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
                    img2 = image.copy()
                    cv2.putText(img2, torso_inclination_text, (10, h - 50), font, 0.9, yellow, 2)
                    cv2.putText(img2, neck_inclination_text, (10, h - 20), font, 0.9, yellow, 2)
                    writer.write(img2)
        

        else:
                soundObj.stop()
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
                cv2.line(image, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
                cv2.line(image, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)
                WRONG_COUNTER = 0
                wrong =0

        
        total_wrong_text = f'Total Wrong Posing: {TOTAL_WRONG}'
        # Timer
        end_time = time.time()-start_time
        end_minutes = int(end_time // 60)
        end_seconds = int(end_time % 60)
        
        
        timer_text = f'Timer: {end_minutes} minutes, {end_seconds} seconds'

        return 1
