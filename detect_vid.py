import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from extra_models.object_detector import ObjectDetector
import matplotlib.pyplot as plt
import argparse
import tqdm
import cv2
import os


parser = argparse.ArgumentParser()
parser.add_argument('--load',dest='load', default=False)
parser.add_argument('-v','--video',dest='video')
parser.add_argument('-f','--frames',dest='frames')
args = parser.parse_args()

video = args.video
video_no_ext = os.path.splitext(video)[0]
print(f'loading video {video}')
cap = cv2.VideoCapture(video)
frames = []
n = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(frame)
        n += 1
        if n % 500 == 0:
            print(f'loading...{n}')
    else:
        break
cap.release()


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


img_size = (640,480)
class_names = ['food']
bbox_sizes = [(25,25)]
backbone_f = 'hr_5_3_8'
intermediate_filters = 256
kernel_size = 16
stride = 8
num_classes = len(class_names) + 1
rfcn_window=3
anchor_ratios = [0.5,1.0,2.0]
anchor_scales = [0.1,0.3,0.6]
test_model = ObjectDetector(
    backbone_f,
    intermediate_filters,
    kernel_size,
    stride,
    img_size,
    num_classes,
    rfcn_window,
    anchor_ratios,
    anchor_scales,
)
load_dir = args.load
test_model.load_weights(load_dir,).expect_partial()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
rpn_writer = cv2.VideoWriter(
    video_no_ext+'_rpn.mp4',
    fourcc,
    10,
    img_size
)
rfcn_writer = cv2.VideoWriter(
    video_no_ext+'_rfcn.mp4',
    fourcc,
    10,
    img_size
)

colors=np.array([
    [255,0,0],
    [0,255,0],
    [255,255,0],
    [0,255,255],
])

t = tqdm.tqdm(unit='frames',total=int(args.frames))
for frame in frames[:int(args.frames)]:
    if np.any(np.not_equal([img_size[1],img_size[0],3],frame.shape)):
        frame = cv2.resize(frame, dsize=img_size)
    float_frame = frame/ 255.0
    rois, rpn_probs, boxes, probs, labels = \
        test_model(float_frame[np.newaxis,...], training=False)
    rpn_frame = frame
    rfcn_frame = frame.copy()
    w, h = np.array(img_size) -1
    for box, p in zip(rois,rpn_probs):
        color = np.clip([4*(1-p),4*(p-0.5),0],0,1)
        x1, y1, x2, y2 = np.multiply(box,[w,h,w,h,]).astype(np.int64)
        rpn_frame[y1,x1:x2] = color
        rpn_frame[y2,x1:x2] = color
        rpn_frame[y1:y2,x1] = color
        rpn_frame[y1:y2,x2] = color

    for box, p, l in zip(boxes, probs, labels):
        color = colors[l] * p
        x1, y1, x2, y2 = np.multiply(box,[w,h,w,h,]).astype(np.int64)
        rfcn_frame[y1,x1:x2] = color
        rfcn_frame[y2,x1:x2] = color
        rfcn_frame[y1:y2,x1] = color
        rfcn_frame[y1:y2,x2] = color
    
    rpn_frame = cv2.cvtColor(rpn_frame,cv2.COLOR_BGR2RGB)
    rfcn_frame = cv2.cvtColor(rfcn_frame,cv2.COLOR_BGR2RGB)
    rpn_writer.write(rpn_frame)
    rfcn_writer.write(rfcn_frame)
    t.update(n=1)
t.close()
rpn_writer.release()
rfcn_writer.release()