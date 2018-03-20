import tensorflow as tf
from keras import backend as K

from bounding_box import Box
from yad2k.models.keras_yolo import yolo_boxes_to_corners
from utils import read_classes, read_anchors, preprocess_image, scale_boxes

class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")


def filter_anchor_boxes(box_confidence, boxes, box_class_probs, threshold=.6):
    """
    Filters YOLO boxes by thresholding on object and class confidence
    Filter any box for which the class "score" is less than a chosen threshold.

    Arguments:

    box_confidence -- tensor of shape (19, 19, 5, 1) - confidence probability that there's some object for each of the 5 boxes
    boxes -- tensor of shape (19, 19, 5, 4) - (bx,by,bh,bw)  for each of the 5 boxes per cell.
    box_class_probs -- tensor of shape (19, 19, 5, 80)- detection probabilities (c1,c2,...c80)  for each of the 80 classes for each of the 5 boxes per cell.
    threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box

    Returns:
    scores -- tensor of shape (None,), containing the class probability score for selected boxes
    boxes -- tensor of shape (None, 4), containing (b_x, b_y, b_h, b_w) coordinates of selected boxes
    classes -- tensor of shape (None,), containing the index of the class detected by the selected boxes

    """

    box_scores = box_confidence * box_class_probs

    box_classes = K.argmax(box_scores, axis=-1)  # returns index
    box_class_scores = K.max(box_scores, axis=-1, keepdims=False)  # returns max value

    # Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = tf.greater(box_class_scores,
                                tf.constant(threshold))  # box_class_scores[box_class_scores < threshold]

    # Apply the mask to scores, boxes and classes

    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes


def non_max_suppression(scores, boxes, classes, max_boxes=10, iou_threshold=0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes

    Arguments:
    scores -- tensor of shape (None,), output of filter_anchor_boxes()
    boxes -- tensor of shape (None, 4), output of filter_anchor_boxes()
    classes -- tensor of shape (None,), output of filter_anchor_boxes()
    max_boxes -- integer, maximum number of predicted boxes you'd like
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (, None), predicted score for each box
    boxes -- tensor of shape (4, None), predicted box coordinates
    classes -- tensor of shape (, None), predicted class for each box

    Note: The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """

    max_boxes_tensor = K.variable(max_boxes, dtype='int32')  # tensor to be used in tf.image.non_max_suppression()
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))  # initialize variable max_boxes_tensor

    # Use tf.image.non_max_suppression() to get the list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes, iou_threshold)

    # Use K.gather() to select only nms_indices from scores, boxes and classes
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes


def evaluate(yolo_outputs, image_shape=(720., 1280.), max_boxes=10, score_threshold=.5, iou_threshold=.5):
    """
    Converts the output of YOLO encoding (yolo_outputs) to predicted boxes along with their scores, box coordinates and classes.

    Arguments:
    yolo_outputs -- output of the encoding model (for image_shape of (608, 608, 3)), contains 4 tensors:
                    box_confidence: tensor of shape (None, 19, 19, 5, 1)
                    box_xy: tensor of shape (None, 19, 19, 5, 2)
                    box_wh: tensor of shape (None, 19, 19, 5, 2)
                    box_class_probs: tensor of shape (None, 19, 19, 5, 80)

    image_shape -- tensor of shape (2,) original image shape
    max_boxes -- integer, maximum number of predicted boxes
    score_threshold -- real value, if [ highest class probability score < threshold], then get rid of the corresponding box
    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering

    Returns:
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    """

    # Retrieve outputs of the YOLO model
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # Convert boxes to be ready for filtering functions
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # Perform Score-filtering with a threshold of score_threshold
    scores, boxes, classes = filter_anchor_boxes(box_confidence, boxes, box_class_probs, threshold=score_threshold)

    # Scale boxes back to original image shape.
    boxes = scale_boxes(boxes, image_shape)

    # Non-max suppression with a threshold of iou_threshold
    scores, boxes, classes = non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    return scores, boxes, classes


def predict_and_draw(sess, image, scores, boxes, classes, yolo_model):
    """
    Runs the graph stored in "sess" to predict boxes and draw bounding boxes

    Arguments:
    sess -- your tensorflow/Keras session containing the YOLO graph
    image - Original Image
    scores -- tensor of shape (None, ), predicted score for each box
    boxes -- tensor of shape (None, 4), predicted box coordinates
    classes -- tensor of shape (None,), predicted class for each box
    yolo_model - pre-trained YOLO Model


    Returns:
    image -- Image with bounding box drawn

    """

    # Preprocess your image
    image_data = preprocess_image(image, model_image_size=(608, 608))

    # Run the session with the correct tensors and choose the correct placeholders in the feed_dict.
    # You'll need to use feed_dict={yolo_model.input: ... , K.learning_phase(): 0})
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],
                                                  feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})

    # Draw bounding boxes on the image file
    box = Box(image, out_scores, out_boxes, out_classes, class_names)
    box.draw_boxes()

    # return out_scores, out_boxes, out_classes
    return image
