import cv2
import numpy as np
from PIL import Image
from keras import backend as K
from keras.models import load_model
import matplotlib.pyplot as plt


from yad2k.models.keras_yolo import yolo_head
from yolo import yolo_eval, predict, class_names, anchors

sess = K.get_session()
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))


def pipeline(image):
    #img = Image.fromarray(image)
    scores, boxes, classes = yolo_eval(yolo_outputs)
    image_pred =  predict(sess, image, scores, boxes, classes,yolo_model)
    #open_cv_image = np.array(pil_img)
    #open_cv_image = open_cv_image[:, :, ::-1].copy()
    return image_pred


if __name__ == '__main__':
    image_path = "test_images/test4.jpg"

    img = cv2.imread(image_path)
    image = pipeline(img)

    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    f, ax = plt.subplots(1, 1, figsize=(15, 8))
    # f, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplot2grid(2, 3, figsize=(15, 8))
    f.tight_layout()

    ax.imshow(image)
    ax.set_title('Drawn', fontsize=15)

    print("")

    #cv2.imshow("preview", image)
    #cv2.waicv2.waitKey(500)


