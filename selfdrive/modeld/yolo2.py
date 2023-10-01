#!/usr/bin/env python3
import cv2
import time
import numpy as np
import ast
import os
import argparse
import json
import onnx
from requests import post
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")

# os.environ["ZMQ"] = "1"
from cereal import messaging
from cereal.visionipc import VisionIpcClient, VisionStreamType
from openpilot.selfdrive.modeld.runners import ModelRunner, Runtime

TOO_CLOSE_SIZE = 30_000
TOO_FAR_SIZE = 15_000
INPUT_SHAPE = (640, 416)
OUTPUT_SHAPE = (1, 16380, 85)
MODEL_PATHS = {
  ModelRunner.THNEED: Path(__file__).parent / 'models/yolov5n_flat.thneed',
  ModelRunner.ONNX: Path(__file__).parent / 'models/yolov5n_flat.onnx'}
half_width = INPUT_SHAPE[0]/2

def xywh2xyxy(x):
  y = x.copy()
  y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
  y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
  y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
  y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
  return y


def non_max_suppression(boxes, scores, threshold):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == scores.shape[0]
  ys1 = boxes[:, 0]
  xs1 = boxes[:, 1]
  ys2 = boxes[:, 2]
  xs2 = boxes[:, 3]
  areas = (ys2 - ys1) * (xs2 - xs1)
  scores_indexes = scores.argsort().tolist()
  boxes_keep_index = []
  while len(scores_indexes):
    index = scores_indexes.pop()
    boxes_keep_index.append(index)
    if not len(scores_indexes):
      break
    ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index], areas[scores_indexes])
    filtered_indexes = set((ious > threshold).nonzero()[0])
    scores_indexes = [
      v for (i, v) in enumerate(scores_indexes)
      if i not in filtered_indexes
    ]
  return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
  # adapted from https://gist.github.com/CMCDragonkai/1be3402e261d3c239a307a3346360506
  assert boxes.shape[0] == boxes_area.shape[0]
  ys1 = np.maximum(box[0], boxes[:, 0])
  xs1 = np.maximum(box[1], boxes[:, 1])
  ys2 = np.minimum(box[2], boxes[:, 2])
  xs2 = np.minimum(box[3], boxes[:, 3])
  intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
  unions = box_area + boxes_area - intersections
  ious = intersections / unions
  return ious


def nms(prediction, conf_thres=0.5, iou_thres=0.45):
  prediction = prediction[prediction[..., 4] > conf_thres]
  boxes = xywh2xyxy(prediction[:, :4])
  res = non_max_suppression(boxes, prediction[:, 4], iou_thres)
  result_boxes = []
  result_scores = []
  result_classes = []
  for r in res:
    result_boxes.append(boxes[r])
    result_scores.append(prediction[r, 4])
    result_classes.append(np.argmax(prediction[r, 5:]))
  return np.c_[result_boxes, result_scores, result_classes]


class YoloRunner:
  def __init__(self):
    onnx_path = MODEL_PATHS[ModelRunner.ONNX]
    if not onnx_path.exists():
      yolo_url = 'https://github.com/YassineYousfi/yolov5n.onnx/releases/download/yolov5n.onnx/yolov5n_flat.onnx'
      os.system(f'wget -P {onnx_path.parent} {yolo_url}')

    model = onnx.load(onnx_path)
    class_names_dict = ast.literal_eval(model.metadata_props[0].value)
    self.class_names = [class_names_dict[i] for i in range(len(class_names_dict))]

    self.output = np.zeros(np.prod(OUTPUT_SHAPE), dtype=np.float32)
    self.model = ModelRunner(MODEL_PATHS, self.output, Runtime.GPU, False, None)
    self.model.addInput("image", None)

  def preprocess_image(self, img):
    img = cv2.resize(img, INPUT_SHAPE)
    img = np.expand_dims(img, 0).astype(np.float32) # add batch dim
    img = img.transpose(0, 3, 1, 2) # NHWC -> NCHW
    img = img[:, [2,1,0], :, :] # BGR -> RGB
    img /= 255 # 0-255 -> 0-1
    return img

  def run(self, img):
    img = self.preprocess_image(img)
    self.model.setInputBuffer("image", img.flatten())
    self.model.execute()
    res = nms(self.output.reshape(1, 16380, 85))
    return [{
        "pred_class": self.class_names[int(opt[-1])],
        "prob": opt[-2],
        "pt1": opt[:2].astype(int).tolist(),
        "pt2": opt[2:4].astype(int).tolist()}
      for opt in res]

  def draw_boxes(self, img, objects):
    img = cv2.resize(img, INPUT_SHAPE)
    for obj in objects:
      pt1 = obj['pt1']
      pt2 = tuple(obj['pt2'])
      cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
      cv2.putText(img, f"{obj['pred_class']} {obj['prob']:.2f}", pt1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return img

def get_yaw(image_center):
  if abs(image_center - half_width) < 10:
    return 0
  offset = half_width - image_center
  power = offset / half_width
  if abs(power) > 1:
    return 1
  return 3.3*power

def in_sweet_spot(size):
  return size > TOO_FAR_SIZE and size < TOO_CLOSE_SIZE
def youre_too_close(size):
  return size < TOO_FAR_SIZE
def wait_for_me(size):
  return size > TOO_CLOSE_SIZE

def get_acceleration(size):
  if in_sweet_spot(size):
    return 0
  if youre_too_close(size):
    return -0.9
  if wait_for_me(size):
    return 0.8
  return .00001
#   return (size - HURRY_UP_SIZE) / (TOO_CLOSE_SIZE - HURRY_UP_SIZE)

def determine_direction(object):
  image_center = (object['pt1'][0] + object['pt2'][0]) / 2
  return get_acceleration(object["size"]), get_yaw(image_center)

def main(debug=False, local=True):
  print("LOCALLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL =",local)
  local = True
  yolo_runner = YoloRunner()
  if local:
    pm = messaging.PubMaster(['customReservedRawData1'])
  # del os.environ["ZMQ"]
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_DRIVER, True)

  while not vipc_client.connect(False):
    time.sleep(0.1)

  last_match = { "size": 0, "last_match": 0 }
  while True:
    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    st = time.time()
    imgff = yuv_img_raw.data.reshape(-1, vipc_client.stride)
    imgff = imgff[:vipc_client.height * 3 // 2, :vipc_client.width]
    img = cv2.cvtColor(imgff, cv2.COLOR_YUV2BGR_NV12)
    outputs = yolo_runner.run(img)
    cv2.imwrite(str(Path(__file__).parent / 'yolo.jpg'), yolo_runner.draw_boxes(img, outputs))

    best_match = { "size": 0, "last_match": 0 }
    for obj in outputs:
      if obj['pred_class'] != "person":
        continue
      size = (obj['pt1'][0] - obj['pt2'][0])*(obj['pt1'][1] - obj['pt2'][1])
      if size < best_match["size"]:
        continue
      best_match = { **obj, "size": size, "last_match": 0 }

    if not best_match["size"] and last_match["size"]:
      best_match = last_match
      best_match["last_match"] += 1

    if best_match["last_match"] > 5:
      continue

    if not best_match["size"]:
      continue

    last_match = best_match
    forward_power, yaw_power = determine_direction(best_match)
      
    if local:
      msg = messaging.new_message()
      msg.customReservedRawData1 = json.dumps({"left": yaw_power, "back": forward_power}).encode()
      pm.send('customReservedRawData1', msg)
    else:
      post("https://192.168.63.84:5000/drive", json={"left": yaw_power, "back": forward_power}, verify=False)
    if debug:
      et = time.time()
      cv2.imwrite(str(Path(__file__).parent / 'yolo.jpg'), yolo_runner.draw_boxes(img, outputs))
      print(f"eval time: {(et-st)*1000:.2f}ms, found: {','.join([x['pred_class'] for x in outputs])}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Receive VisionIPC frames, run YOLO and publish outputs")
  parser.add_argument("--debug", action="store_true", help="debug output")
  parser.add_argument("--local", action="store_true", help="run locally")
  args = parser.parse_args()

  main(debug=args.debug, local=args.local)
