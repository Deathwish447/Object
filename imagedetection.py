import numpy as np
import argparse
from matplotlib import pyplot as plt
import time
import cv2
from pixellib.torchbackend.instance import instanceSegmentation
import os
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,help="path to input image")
ap.add_argument("-c", "--confidence", type=float, default=0.5,help="minimum probability to filter weak detections, IoU threshold")
ap.add_argument("-t", "--threshold", type=float, default=0.3,help="threshold when applying non-maxima suppression")
args = vars(ap.parse_args())
labelsPath = 'yolo-coco\\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
weightsPath = 'yolo-coco\\yolov3.weights'
configPath = 'yolo-coco\\yolov3.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]
ln = net.getLayerNames()
ln = [ln[i - 1]  for i in net.getUnconnectedOutLayers()]
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
net.setInput(blob)
layerOutputs = net.forward(ln)
boxes = []
confidences = []
classIDs = []
for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]
		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")
			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))
			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])
if len(idxs) > 0:
	for i in idxs.flatten():
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)
ins = instanceSegmentation()
ins.load_model("pointrend_resnet50.pkl")
# ins.select_target_classes(person=True)
ar=args["image"].split("/")
ins.segmentImage(args["image"],show_bboxes=True,output_image_name=ar[2])
img = cv2.imread(args["image"])


grab = cv2.imread(args["image"])
mask = np.zeros(grab.shape[:2], np.uint8)
backgroundModel = np.zeros((1, 65), np.float64)
foregroundModel = np.zeros((1, 65), np.float64)
rectangl = (x, y,
			 w, h)
cv2.grabCut(grab, mask, rectangl,
            backgroundModel, foregroundModel,
            3, cv2.GC_INIT_WITH_RECT)
mask2 = np.where((mask == 2)|(mask == 0), 0, 1).astype('uint8')
grab = grab * mask2[:, :, np.newaxis]
cv2.imshow("GrabCut",grab)

img2 = cv2.imread(ar[2])
cv2.imshow("Input",img)
cv2.imshow("Detection", image)
cv2.imshow("Output",img2)
cv2.waitKey(0)