import tensorflow_hub as hub
import tensorflow as tf
import cv2


detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite2/detection/1")

def inference_from_tf_model(frame):
    # Convert img to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Is optional but i recommend (float convertion and convert img to tensor image)
    rgb_tensor = tf.convert_to_tensor(rgb, dtype=tf.uint8)

    # Add dims to rgb_tensor
    rgb_tensor = tf.expand_dims(rgb_tensor, 0)

    boxes, scores, classes, num_detections = detector(rgb_tensor)

    pred_labels = classes.numpy().astype('int')[0]

    with open('/home/tiago/Projetos/python-utils/core/video/Capture/core/models/mask_rcnn_inception_v2_coco_2018_01_28/object_detection_classes_coco.txt', 'r') as file:
        labels = file.read().strip().split("\n")

    pred_labels = [labels[i] for i in pred_labels]
    pred_boxes = boxes.numpy()[0].astype('int')
    pred_scores = scores.numpy()[0]

    # loop throughout the detections and place a box around it
    for score, (ymin, xmin, ymax, xmax), label in zip(pred_scores, pred_boxes, pred_labels):
        if score < 0.5:
                continue

        score_txt = f'{100 * round(score,0)}'
        img_boxes = cv2.rectangle(
            rgb, (xmin, ymax), (xmax, ymin), (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img_boxes, label, (xmin, ymax-10),
                    font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img_boxes, score_txt, (xmax, ymax-10),
                    font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

        frame = img_boxes

    return frame