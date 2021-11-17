import numpy as np
import time
import os
import cv2
import glob

def pred(img):
  # load the COCO class labels our YOLO model was trained on
    labelsPath = "../yoloV4-ppe-detection-model/obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")
    # derive the paths to the YOLO weights and model configuration
    weightsPath = '../yoloV4-ppe-detection-model/training/yolov4-custom_last.weights'
    configPath = '../yoloV4-ppe-detection-model/yolov4-custom.cfg'
    # load our YOLO object detector trained on COCO dataset (80 classes)
    # print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    image = cv2.imread('../yoloV4-ppe-detection-model/test/'+img)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
    # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # print('classIDs',classIDs)
    for k in classIDs:
        if(k == 0) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 1) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 2) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.15,0.1)
        elif(k == 3) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8,0.1)
        elif(k == 4) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 5) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.8,0.1)
        elif(k == 6) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 7) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 8) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 9) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 10) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        elif(k == 11) :
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35,0.1)
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,0.5, color, 2)
        return image


path = '../yoloV4-ppe-detection-model/test/'
extension = 'jpg'
os.chdir(path)
bmp_files = glob.glob('*.{}'.format(extension))

start = time.time()
for name in bmp_files:
    try:
        image = pred(name)
        cv2.imwrite("../yoloV4-ppe-detection-model/output/"+name, image)
    except:
        try:
            image = cv2.imread('../yoloV4-ppe-detection-model/test/'+name)
            cv2.imwrite("../yoloV4-ppe-detection-model/output/not_detect/"+name, image)
        except:
            pass
        pass
print('time for evaluation is =',time.time()-start,'seconds')
print('evaluation complete')