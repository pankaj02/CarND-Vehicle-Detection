import glob

import cv2
from keras import backend as K
from keras.models import load_model

from yad2k.models.keras_yolo import yolo_head
from yolo import evaluate, predict_and_draw, class_names, anchors

sess = K.get_session()
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


def pipeline(image):
    scores, boxes, classes = evaluate(yolo_outputs)
    image_pred = predict_and_draw(sess, image, scores, boxes, classes, yolo_model)
    return image_pred


if __name__ == '__main__':
    images = glob.glob('test_images/test3*.jpg')

    for image_path in images:
        img = cv2.imread(image_path)
        image = pipeline(img)
        name = image_path.split("\\")[1].split('.')[0]+'.jpg'
        cv2.imwrite('output_images/' + name, image)
