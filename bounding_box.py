import cv2
import numpy as np


class Box:
    def __init__(self, image, out_scores, out_boxes, out_classes, class_names, color=(255, 153, 0)):
        self.image = image
        self.out_scores = out_scores
        self.out_boxes = out_boxes
        self.out_classes = out_classes
        self.class_names = class_names
        self.color = color

    def draw_boxes(self):

        thickness = (self.image.shape[1] + self.image.shape[0]) // 300

        for i, c in reversed(list(enumerate(self.out_classes))):
            predicted_class = self.class_names[c]
            box = self.out_boxes[i]
            score = self.out_scores[i]
            label = '{} {:.2f}'.format(predicted_class, score)
            label_size = [100, 40]

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(self.image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(self.image.shape[1], np.floor(right + 0.5).astype('int32'))

            cv2.rectangle(self.image, (left, top), (right, bottom), self.color, thickness)

            if top - label_size[1] >= 0:
                label_top_left = np.array([left - int(thickness / 2), top - label_size[1]])
                text_origin = np.array([left, top - int(label_size[1] / 2)])
            else:
                label_top_left = np.array([left, top + 1])
                text_origin = np.array([left, top - int(label_size[1] / 2)])

            cv2.rectangle(self.image, tuple(label_top_left), tuple(label_top_left + label_size), self.color, -1)
            cv2.putText(self.image, label, tuple(text_origin), cv2.FONT_HERSHEY_DUPLEX, .5, (0, 0, 0), 2)
