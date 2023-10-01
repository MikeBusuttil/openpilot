#!/usr/bin/env python3
import cv2, face_recognition, json, time, os
from requests import post
from cereal import messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType

import warnings
warnings.filterwarnings("ignore")

os.environ["ZMQ"] = "1"
# me = cv2.imread("./yoo/2.5tiles.png")
# me = cv2.cvtColor(me, cv2.COLOR_BGR2RGB)
# top, right, bottom, left = face_recognition.face_locations(me)[0]
# cv2.imshow('face', me[top:bottom, left:right])
# cv2.waitKey(0)
me2 = []
for n in [75, 107, 108, 155, "1tile"]:
    me = cv2.imread(f"./yoo/{n}.png")
    me = face_recognition.face_encodings(me)[0]
    me2.append(me)

# face_height_pixels = bottom - top
# face_width_pixels = right - left
# print("H x W", face_height_pixels, face_width_pixels)
# face_width_inches = 5.5
# face_height_inches = 8.5
# tile length = 36"

height = 1208
half_width = 1928/2
max_size = 120
min_size = 40
size_range = max_size - min_size

def locate_match(frame):
    locations = face_recognition.face_locations(frame)
    for top, right, bottom, left in locations:
        cv2.imwrite(f"{right - left}.png", frame)
        face = face_recognition.face_encodings(frame, known_face_locations=[(top, right, bottom, left)])[0]
        if face_recognition.compare_faces(me2, face):
            return (top, right, bottom, left)
        else:
            print('rejected')
        
def calculate_distance(top, right, bottom, left):
    '''
    TODO:
      - take lots of measurements to determine distance
      - use intrinsic matrix to un-distort the image first
    '''
    return None

def get_yaw(image_center):
    offset = half_width - image_center
    power = offset / half_width
    return min(power, 1)

def get_acceleration(size):
    if size > max_size:
        return 0
    if size < min_size:
        return 1
    return (size - min_size) / size_range

def main_loop(frame, pm):
    location = locate_match(frame)
    if not location:
        #TODO don't just full stop if you don't see me
        # might also be good for CPU utilization to go with the current trajectory for a couple frames without processing
        # good UX might be to average over the last 5 (or so) frames
        return
    
    _, right, _, left = location
    center = (right + left) / 2
    size = right - left

    yaw_power = get_yaw(center)
    forward_power = get_acceleration(size)

    msg = messaging.new_message()
    # post("https://192.168.63.84:5000/drive", json={"back": forward_power, "left": yaw_power}, verify=False)
    post("https://192.168.63.84:5000/drive", json={"left": yaw_power, "back": 0}, verify=False)
    msg.customReservedRawData1 = json.dumps({"back": forward_power, "left": yaw_power}).encode()
    print(f"size={size}  center={center}")
    # print(f"move back={forward_power} left={yaw_power}")
    # pm.send('customReservedRawData1', msg)

def main():
    # pm = messaging.PubMaster(['customReservedRawData1'])
    pm = None
    del os.environ["ZMQ"]
    vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)

    while not vipc_client.connect(False):
        time.sleep(0.1)
    while True:
        yuv_img_raw = vipc_client.recv()
        if yuv_img_raw is None or not yuv_img_raw.data.any():
            continue
        frame = yuv_img_raw.data.reshape(-1, vipc_client.stride)
        frame = frame[:vipc_client.height * 3 // 2, :vipc_client.width]
        frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
        main_loop(frame, pm)

if __name__ == "__main__":
    main()
