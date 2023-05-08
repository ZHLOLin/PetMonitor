import io
import RPi.GPIO as GPIO
import email_sender
import threading
import time
import math
from base64 import b64encode
from flask import Flask, render_template, jsonify, request, Response
import atexit
import os
import openai
# from picamera import PiCamera
import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision
from datetime import datetime

app = Flask(__name__)

camera = cv2.VideoCapture(0)
# Initialize the object detection model
base_options = core.BaseOptions(
    file_name='lite-model_ssd_mobilenet_v1_1_metadata_2.tflite', use_coral=False, num_threads=4)
detection_options = processor.DetectionOptions(
    max_results=2, score_threshold=0.5)
options = vision.ObjectDetectorOptions(
    base_options=base_options, detection_options=detection_options)
detector = vision.ObjectDetector.create_from_options(options)

# Set up the GPIO pins for the servo motor
servo_pin = 18
GPIO.setmode(GPIO.BCM)
GPIO.setup(servo_pin, GPIO.OUT)


# Set up some global variable for the server
last_remind = time.time()
last_seen = time.time()
last_eat = time.time()
last_drink = time.time()
prev_center = None
eat_count = 0
drink_count = 0
eat_time = 0
drink_time = 0
servo_moving = False

# Initialize the servo motor
servo = GPIO.PWM(servo_pin, 50)
servo.start(0)



def gen_frames():
    while True:
        ret, image = camera.read()
        detect(image)
        if not ret:
            break
        # if detect_cat(image):
        #     print("Cat found")
        ret, buffer = cv2.imencode('.jpg', image)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def spin_servo(servo, start_angle, end_angle, delay):
    global servo_moving
    while servo_moving:
        for angle in range(start_angle, end_angle + 1):
            #print("-")
            if not servo_moving:
                return
            duty_cycle = angle / 18 + 2
            servo.ChangeDutyCycle(duty_cycle)
            time.sleep(delay)
        for angle in range(end_angle, start_angle - 1, -1):
            #print("-")
            if not servo_moving:
                return
            duty_cycle = angle / 18 + 2
            servo.ChangeDutyCycle(duty_cycle)
            time.sleep(delay)
            
def check_cat_behaviour(cat_bbox, target_bbox, threshold=0.2):
    cat_xmin, cat_ymin, cat_xmax, cat_ymax = cat_bbox
    target_xmin, target_ymin, target_xmax, target_ymax = target_bbox
    print(cat_bbox)

    # Calculate the area of intersection
    intersection_area = max(0, min(cat_xmax, target_xmax) - max(cat_xmin, target_xmin)) * max(0, min(cat_ymax, target_ymax) - max(cat_ymin, target_ymin))

    cat_area = (cat_xmax - cat_xmin) * (cat_ymax - cat_ymin)
    target_area = (target_xmax - target_xmin) * (target_ymax - target_ymin)
    union_area = cat_area + target_area - intersection_area
    iou = intersection_area / union_area

    # Check if the cat is drinking or eating
    print(iou)
    return iou >= threshold

def detect(image):
    global last_remind, last_seen, prev_center
    global eat_count, eat_time
    global drink_count, drink_time
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (300, 300))
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Create a TensorImage object from the RGB image.
    input_tensor = vision.TensorImage.create_from_array(rgb_image)
    # Run object detection estimation using the model.
    detection_result = detector.detect(input_tensor)
    cat_found = False
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        if category_name == "cat":
            cat_found = True
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            print(start_point, end_point)
            cat_bbox = bbox.origin_x, bbox.origin_y, bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            food_bbox = (182, 184, 283, 277)
            water_bbox = (123, 213, 223, 285)
            if check_cat_behaviour(cat_bbox, water_bbox):
                drink_count += 1
                if drink_count >= 30:
                    drink_time += 1
                    drink_count = 0
            if check_cat_behaviour(cat_bbox, food_bbox):
                eat_count += 1
                if eat_count >= 45:
                    eat_time += 1
                    eat_count = 0
            center = (start_point[0] + end_point[0]) // 2, (start_point[1] + end_point[1]) // 2
            success, encoded_image = cv2.imencode(".png", image)
            if not success:
                raise ValueError("Failed to encode the image")
            img_data = encoded_image.tobytes()
            if prev_center is None:
                prev_center = center
            else:
                distance = math.sqrt((center[0] - prev_center[0]) ** 2 + (center[1] - prev_center[1]) ** 2)
                speed = distance / (time.time() - last_seen)
                last_seen = time.time()
                print("speed" + str(speed))
                if speed >= 9.5:
                    email_sender.send_mail(1, img_data)
                prev_center = center
            if time.time() - last_remind >= 30:
                email_sender.send_mail(0, img_data)
                last_remind = time.time()
                print("cat" + str(time.time()))
    if not cat_found:
        prev_center = None


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form.get('question')
    print(question)
    response = ask_chatgpt(question)
    return jsonify({'response': response})

def ask_chatgpt(question):
    model_engine = "text-davinci-002"
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"{question}\n"

    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.5,
    )

    message = response.choices[0].text.strip()
    return message

@app.route('/control_teasing', methods=['POST'])
def control_teasing():
    switch = request.form.get('value')
    global servo_moving
    if switch == "0":
        print("stop")
        print(switch)
        servo_moving = False
        servo.ChangeDutyCycle(0)
    else:
        print("start")
        print(switch)
        servo_moving = True
        start_angle = 0
        end_angle = 45
        delay = 0.005
        t = threading.Thread(target=spin_servo, args=(servo, start_angle, end_angle, delay))
        t.start()
    return jsonify({'value': switch})

@app.route('/get_statistics', methods=['GET'])
def get_statistics():
    # Send the statistic to the frontend webpage
    stat1 = eat_time
    stat2 = drink_time
    
    return jsonify({'stat1': stat1, 'stat2': stat2})
    
def cleanup(exception=None):
    servo.stop()
    GPIO.cleanup()
    print("Cleaning up resources...")

atexit.register(cleanup)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
