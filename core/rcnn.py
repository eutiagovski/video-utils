import numpy as np
import time
import cv2


def check_masks(frame):
    # load labels
    LABELS = open("./models/mask_rcnn_inception_v2_coco_2018_01_28/object_detection_classes_coco.txt").read().strip().split("\n")
    LABELS

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")

    # load our Mask R-CNN trained on the COCO dataset (90 classes)
    # from disk

    # derive the paths to the Mask R-CNN weights and model configuration
    weightsPath = "./models/mask_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb"
    configPath = "./models/mask_rcnn_inception_v2_coco_2018_01_28/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

    print("[INFO] loading Mask R-CNN from disk...")
    net = cv2.dnn.readNetFromTensorflow(weightsPath, configPath)

    blob = cv2.dnn.blobFromImage(frame, swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    (boxes, masks) = net.forward(["detection_out_final", "detection_masks"])
    end = time.time()

    for i in range(0, boxes.shape[2]):
        class_id = int(boxes[0, 0, i, 1])
        confidence = boxes[0, 0, i, 2]

        if confidence > 0.5:
            (H, W) = frame.shape[:2]

            box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H]) 
            (startX, startY, endX, endY) = box.astype("int")
            boxW = endX - startX
            boxH = endY - startY

            mask = masks[i, class_id]
            mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0.3)

            roi = frame[startY:endY, startX:endX][mask]
        
            color = COLORS[class_id]
            blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

            frame[startY:endY, startX:endX][mask] = blended

            color = [int(c) for c in color]
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

            text = f"{LABELS[class_id]}: {confidence:.4f}"
            cv2.putText(frame, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame