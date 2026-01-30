import copy
import csv
import itertools
from collections import deque
from typing import List
import pyttsx3
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from flask import Flask, request, Response
from flask_cors import CORS
import wordninja
import datetime
import time
import io
import base64

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)  
tts_engine.setProperty('volume', 1.0)
app = Flask(__name__)
CORS(app)

# Global variable to store current frame
current_frame = None
frame_lock = threading.Lock()
class KeyPointClassifier(object):
    def __init__(self, model_path='model/slr_model.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32)
        )
        self.interpreter.invoke()
        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        
        if max(np.squeeze(result)) > 0.5:
            result_index = np.argmax(np.squeeze(result))
            return result_index
        else:
            return 25

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick
        self._difftimes.append(different_time)
        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        return round(fps, 2)


def draw_landmarks(image, landmark_point):
    neon_green = (20, 255, 57)
    neon_purple = (253, 18, 171)
    neon_red = (49, 49, 255)
    neon_blue = (255, 81, 31)
    neon_orange = (31, 91, 255)
    yellow = (0, 255, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    
    if len(landmark_point) > 0:
        #Thumb
        skeletal_color = neon_green
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), black, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), black, 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]), skeletal_color, 2)

        # Index finger
        skeletal_color = neon_purple
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), black, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), black, 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), black, 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]), skeletal_color, 2)

        # Middle finger
        skeletal_color = neon_red
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), black, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), black, 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), black, 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]), skeletal_color, 2)

        # Ring finger
        skeletal_color = neon_blue
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), black, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), black, 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), black, 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]), skeletal_color, 2)

        # Little finger
        skeletal_color = neon_orange
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), black, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), black, 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), black, 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]), skeletal_color, 2)
        
        # Palm
        skeletal_color = yellow
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), black, 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), black, 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), black, 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), black, 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), black, 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), black, 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]), skeletal_color, 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), black, 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]), skeletal_color, 2)
        
        skeletal_color = white

    for index, landmark in enumerate(landmark_point):
        radius = 8 if index in [4, 8, 12, 16, 20] else 5
        cv.circle(image, (landmark[0], landmark[1]), radius, skeletal_color, -1)
        cv.circle(image, (landmark[0], landmark[1]), radius, black, 1)

    return image


def calc_bounding_rect(image, landmarks) -> List:
    image_width = image.shape[1]
    image_height = image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x+w, y+h]


def calc_landmark_list(image, landmarks) -> List:
    image_width = image.shape[1]
    image_height = image.shape[0]
    landmark_point = []

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list) -> List:
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]
        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert into a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization (-1 to 1)
    max_value = max(list(map(abs, temp_landmark_list)))
    if max_value > 0:
        temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list


def draw_bounding_rect(image, use_brect, brect, outline_color=(255, 255, 255), pad=6):
    if use_brect:
        cv.rectangle(
            image,
            (brect[0] - pad, brect[1] - pad),
            (brect[2] + pad, brect[3] + pad),
            outline_color,
            1
        )
    return image


def draw_hand_label(image, brect, handedness):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22), (0, 0, 0), -1)
    hand = handedness.classification[0].label[0:]
    cv.putText(image, hand, (brect[0] + 5, brect[1] - 4), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def get_result_image():
    return np.ones([127, 299, 3], dtype=np.uint8) * 255


def get_fps_log_image():
    return np.ones([30, 640, 3], dtype=np.uint8) * 255


def show_result(image, handedness, hand_sign_text):
    if hand_sign_text != "":
        hand = handedness.classification[0].label[0:]
        position = (10, 80)
        if hand == "Right":
            cv.putText(image, hand_sign_text, position, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 6, cv.LINE_AA)
        elif hand == "Left":
            cv.putText(image, "SPACE", position, cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 6, cv.LINE_AA)
    return image


def show_fps_log(image, fps, log=""):
    cv.putText(image, str(fps), (0, 22), cv.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 0), 1, cv.LINE_AA)
    if log != "":
        cv.putText(image, log, (90, 22), cv.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 0), 1, cv.LINE_AA)
    return image


import threading

@app.route('/video_feed', methods=['POST'])
def receive_video():
    global current_frame
    try:
        # Get image data from request
        image_data = request.data
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
        
        return Response(status=200)
    except Exception as e:
        print(f"Error receiving video: {e}")
        return Response(status=500)

@app.route('/get_frame')
def get_frame():
    global current_frame
    with frame_lock:
        if current_frame is not None:
            # Encode frame to JPEG
            _, buffer = cv.imencode('.jpg', current_frame)
            frame_bytes = buffer.tobytes()
            return Response(frame_bytes, mimetype='image/jpeg')
    
    # Return blank frame if no frame available
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv.imencode('.jpg', blank_frame)
    frame_bytes = buffer.tobytes()
    return Response(frame_bytes, mimetype='image/jpeg')

def main():
    global current_frame
    print("INFO: Initializing System")
    output_file = "recognized_signs.txt"
    last_sign = ""
    sign_buffer = []
    sign_stable_count = 0
    STABILITY_THRESHOLD = 10
    current_text = ""
    
    CAP_DEVICE = 0
    CAP_WIDTH = 640
    CAP_HEIGHT = 480
    FRAME_DELAY = 0.1
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    print("INFO: Waiting for ESP32 camera feed...")
    print("Server running on http://0.0.0.0:5000")
    print("ESP32 should send video to http://YOUR_PC_IP:5000/video_feed")
    
    # Load background image
    background_template = cv.imread("resources/background.png")
    
    # Setup MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE
    )
    
    # Load model
    keypoint_classifier = KeyPointClassifier()
    
    # Load labels
    keypoint_labels_file = "model/label.csv"
    with open(keypoint_labels_file, encoding="utf-8-sig") as f:
        key_points = csv.reader(f)
        keypoint_classifier_labels = [row[0] for row in key_points]
    
    # Start Flask server in a separate thread
    server_thread = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=5000, debug=False, threaded=True))
    server_thread.daemon = True
    server_thread.start()
    
    # FPS calculator
    cv_fps = CvFpsCalc(buffer_len=10)
    
    # Give server time to start
    time.sleep(2)
    print("INFO: System is up & running")
    while True:
        fps = cv_fps.get()
        
        # Check for exit
        key = cv.waitKey(1)
        if key == 27:  # ESC key
            print("INFO: Exiting...")
            break
        elif key == 8 or key == 255:  # Delete/Backspace
            if current_text:
                current_text = current_text[:-1]
                print(f"Deleted. Current text: '{current_text}'")
                # Update file
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(current_text)
        elif key == 32:  # Spacebar - speak text
            if current_text:
                print(f"Speaking: '{current_text}'")
                tts_engine.say(current_text)
                tts_engine.runAndWait()
        
        # Get frame from ESP32
        with frame_lock:
            if current_frame is not None:
                image = current_frame.copy()
            else:
                # Create blank frame if no feed
                image = np.zeros((CAP_HEIGHT, CAP_WIDTH, 3), dtype=np.uint8)
                cv.putText(image, "Waiting for ESP32...", (50, 240), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                debug_image = image.copy()
                result_image = get_result_image()
                fps_log_image = get_fps_log_image()
                
                # Show waiting screen
                if background_template is not None:
                    background_image = background_template.copy()
                    background_image[170:170 + 480, 50:50 + 640] = debug_image
                    background_image[240:240 + 127, 731:731 + 299] = result_image
                    background_image[678:678 + 30, 118:118 + 640] = fps_log_image
                    cv.imshow("Sign Language Recognition", background_image)
                else:
                    cv.imshow("Sign Language Recognition", debug_image)
                
                time.sleep(FRAME_DELAY)
                continue
        
        image = cv.resize(image, (CAP_WIDTH, CAP_HEIGHT))
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)
        result_image = get_result_image()
        fps_log_image = get_fps_log_image()
        
        # Convert BGR to RGB for MediaPipe
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        
        # Show FPS
        fps_log_image = show_fps_log(fps_log_image, fps)
        
        # Process hand detection
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Calculate bounding box
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                
                # Calculate landmarks
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                
                # Preprocess landmarks
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                
                if hand_sign_id == 25:
                    hand_sign_text = ""
                else:
                    hand_sign_text = keypoint_classifier_labels[hand_sign_id]
                
                # Check if left hand for space
                hand = handedness.classification[0].label[0:]
                if hand == "Left" and hand_sign_id != 25:
                    # Left hand - add space
                    current_text += " "
                    print(f"Space added. Current text: '{current_text}'")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(current_text)
                    # Speak the current text
                    if current_text.strip():
                        tts_engine.say(current_text)
                        tts_engine.runAndWait()
                elif hand == "Right" and hand_sign_text and hand_sign_text == last_sign:
                    sign_stable_count += 1
                    if sign_stable_count == STABILITY_THRESHOLD:
                        sign_buffer.append(hand_sign_text)
                        current_text += hand_sign_text
                        print(f"Detected: {hand_sign_text}. Current text: '{current_text}'")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(current_text)
                        # Speak current text
                        tts_engine.say(current_text)
                        tts_engine.runAndWait()
                else:
                    last_sign = hand_sign_text
                    sign_stable_count = 0
                result_image = show_result(result_image, handedness, hand_sign_text)
                
                debug_image = draw_bounding_rect(debug_image, True, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_hand_label(debug_image, brect, handedness)
        
        if background_template is not None:
            background_image = background_template.copy()
            background_image[170:170 + 480, 50:50 + 640] = debug_image
            background_image[240:240 + 127, 731:731 + 299] = result_image
            background_image[678:678 + 30, 118:118 + 640] = fps_log_image
            
            # Display current text
            text_display = f"Text: {current_text}"
            if len(text_display) > 60:
                text_display = text_display[:57] + "..."
            cv.putText(background_image, text_display, (50, 650), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
            
            cv.imshow("Sign Language Recognition", background_image)
        else:
            cv.imshow("Sign Language Recognition", debug_image)
    
        time.sleep(FRAME_DELAY)

def split_if_needed(word):
    if word.isalpha() and len(word) > 6:
        return " ".join(wordninja.split(word))
    return word

with open("recognized_signs.txt") as f:
    for line in f:
        words = line.split()
        fixed = [split_if_needed(w) for w in words]
        print(" ".join(fixed))
    cap.release()
    cv.destroyAllWindows()
    print("INFO: Bye")


if __name__ == "__main__":
    main()

