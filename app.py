import cv2
import mediapipe as mp
import numpy as np

pose_model = mp.solutions.pose
pose = pose_model.Pose()

jump_count = 0
is_jumping = False
previous_foot_y = None

COLOR_RANGES = {
    'red': ([0, 120, 70], [10, 255, 255]),
    'blue': ([100, 150, 50], [140, 255, 255]),
    'green': ([40, 40, 40], [80, 255, 255]),
    'yellow': ([20, 100, 100], [30, 255, 255]),
}

def get_color_bound(color):

    if color in COLOR_RANGES:
        lower_limit, upper_limit = COLOR_RANGES[color]
        return np.array(lower_limit), np.array(upper_limit)
    else:
        print('Defaulting to color Red')
        lower_limit, upper_limit = COLOR_RANGES['red']
        return np.array(lower_limit), np.array(upper_limit)

def rope_detection(frame, lower_limit, upper_limit):
    
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    return mask

def rope_count_processing(source = 'webcam', video_path = None, color= 'red'):

    global jump_count, is_jumping, previous_foot_y 

    if source == 'webcam':
        capture = cv2.VideoCapture(0)
    elif source == 'video' and video_path:
        capture = cv2.VideoCapture(video_path)
    else:
        print('Unexpected method try a different method (webcam or video) DO NOT forget to upload the video')
        return
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break

        lower_limit, upper_limit = get_color_bound(color)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb_frame)

        rope = rope_detection(frame, lower_limit, upper_limit)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            left_ankle = landmarks[pose_model.PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[pose_model.PoseLandmark.RIGHT_ANKLE]

            foot_y = (left_ankle.y + right_ankle.y) / 2

            rope_below_feet = np.any(rope[int(foot_y * frame.shape[0]) - 10:int(foot_y * frame.shape[0]) + 10, :])

            if rope_below_feet and not is_jumping:
                jump_count += 1
                is_jumping = True
            elif not rope_below_feet:
                is_jumping = False


            cv2.putText(frame, f'Jumps: {jump_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Jump Rope Counter", frame)
        #cv2.imshow("Rope Detection", rope)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

rope_count_processing('video','jump-rope-counter-test.mov','green')
        