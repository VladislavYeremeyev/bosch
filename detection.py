"""
    File contains function for detection structure
"""

"""
    installing TensorFlow (system functions):
    !apt-get install protobuf-compiler python-pil python-lxml python-tk
    !pip install Cython
    !git clone https://github.com/tensorflow/models.git
    !cd models/research; protoc object_detection/protos/*.proto --python_out=.
    !cd models/research; export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim; python object_detection/builders/model_builder_test.py
"""

from eqs import idEq, trueEq

#for detection
import sys, os

sys.path.append('models/research')
sys.path.append('models/research/object_detection')

import numpy as np
import six.moves.urllib as urllib
import tarfile
import zipfile
from PIL import Image
from matplotlib import pyplot as plt
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from utils import label_map_util
from utils import visualization_utils as vis_util

model_path = 'http://download.tensorflow.org/models/object_detection/'
model_name = 'ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03'
"""
    for detection with masks (segments) use this model:
    model_name = 'mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28'
"""

os.environ['MODEL_PATH'] = model_path + model_name + '.tar.gz'
os.environ['MODEL_FILE_NAME'] = model_name + '.tar.gz'


"""
system functions (with '!' sign)
!rm $MODEL_FILE_NAME
!wget $MODEL_PATH
!tar xfz $MODEL_FILE_NAME
"""


#load model in memory
model_file_name =  model_name + '/frozen_inference_graph.pb'
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(model_file_name, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

        
#load class label
label_map = label_map_util.load_labelmap('models/research/object_detection/data/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories) 
# dictionary with class ids, 
#e. g. {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, ...}


#load image method
def load_image(image_file_name):
    image = Image.open(image_file_name)
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


#prepare detection raw data
def run_inference_for_single_image_raw(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            if 'detection_masks' in tensor_dict:
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # searching objects
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
      
    return output_dict


def run_inference_for_single_image(output_dict):
    # transform float32 arrays in usable format
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
        
    return output_dict


class StructType:
    def __init__(self, sType, sCategory):
        """
            Define struct  type


            Parameters:
            -----------------
            sType : type?
                Type of structure
            sCategory : type?
                Category of structure
        """
        self.sType = sType
        self.sCategory = sCategory

class Structure:

    def __init__(self, obj, coord, structType):
        """
            Define structure

            Parameters:
            -----------------
            obj : type?
                Visual representation of structure
            coord : type?
                Structure coordinate on the image
            structType : StructType
                Type of structure
        """

        self.obj        = obj
        self.coord      = coord
        self.structType = structType


def detectStructures(imagePath):
    """
        Get all structures from an image

        Parameters:
        --------------------
        imagePath : type?
            Path to the folder with images

        Return:
        --------------------
        structures : collection of structures
            Set of structure from the images
    """
    img = load_image(imagePath)
    
    raw_data = run_inference_for_single_image_raw(img, detection_graph)
    output_dict = run_inference_for_single_image(raw_data)

    return output_dict


def visualize_detected_objects(imagePath):
    img = load_image(imagePath)
    
    vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks = output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    
    plt.figure(figsize=(12, 8))
    plt.grid(False)
    plt.imshow(img)
    
    
def create_objects_dictionary(output_dict, type_dictionary):
     """
        Creates dictionary with detected objects properties

        Parameters:
        --------------------
        output_dict : dictionary
            result of detectStructures function invoke
        type_dictionary : dictionary
            dictionary with types of objects in model (e. g. {{'id': 1, 'name': 'person'}, ...})

        Return:
        --------------------
        result_dictionary : dictionary with detected objects properties
    """
    objects_amount = 0
    for i in range(100):
        if output_dict['detection_scores'][i] > 0.5:
            objects_amount += 1
      
    result_dictionary = {}
    for i in range(objects_amount):
        result_dictionary[i] = {
            'type': type_dictionary[output_dict['detection_classes'][i]]['name'],
            'pos': {
                'left': output_dict['detection_boxes'][i][1],
                'top': output_dict['detection_boxes'][i][0]
            },
            'size': {
                'width': output_dict['detection_boxes'][i][3] - output_dict['detection_boxes'][i][1],
                'height': output_dict['detection_boxes'][i][2] - output_dict['detection_boxes'][i][0]
            }
        }
    
  return result_dictionary


def insert_images(image_size, detected_objects_dictionary):
    '''
        imports
        from PIL import Image
        from matplotlib import pyplot as plt
    '''
    img_w, img_h = image_size
    background = Image.new('RGBA', (img_w, img_h), (255, 255, 255, 255))  #white background
    for key, value in detected_objects_dictionary.items():
        offset = (round(value['pos']['left'] * img_w), round(value['pos']['top'] * img_h))
        
        # choose_image function return path of equivalent in new reality object's image
        img_to_insert_path = choose_image(value['type'])
        inserted_img = Image.open(img_to_insert_path, 'r')
        available_width = round(value['size']['x'] * img_w)  # width of available space in bounding box in pixels
        available_height = round(value['size']['y'] * img_h)    # height of available space in bounding box in pixels
        
        #ratio for changing inserted image size to insert it in bounding box area
        ratio = (available_width / float(inserted_img.size[0]))
        proportional_height = int((float(inserted_img.size[1]) * float(ratio)))
        inserted_img = inserted_img.resize((available_width, available_height), Image.ANTIALIAS)  #img resizing

        background.paste(inserted_img, offset)  #insert img into bounding box place
    background.save('out.jpg')
    return background


def detectRealityStructures(folderPath):
    """
        Get all structures from all the images from the given folder

        Parameters:
        --------------------
        folderPath : type?
            Path to the folder with images

        Return:
        --------------------
        structures : collection of structures
            Set of structure from the images
    """

    return None

def isEqual(eqFun, a, b):
    """
        Check equals of structure depend on given equality relation  

        Parameters:
        ------------------
        eqFun : function(a,b) -> bool
            Equality relation
        a : type?
            Structure for checked for equvivalence
        b : type?
            Structure for checked for equvivalence

        Return:
        --------------
        res : bool
            States equvivalence types "a" and "b" or not
    """

    return eqFun(a.structType, b.structType)

