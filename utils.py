from . import hparams_config
import numpy as np
from typing import Text, Tuple, Union
import tensorflow as tf
import yaml

## For visualization

coco = {
    # 0: 'background',
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush',
}

voc = {
    # 0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor',
}

waymo = {
    # 0: 'background',
    1: 'vehicle',
    2: 'pedestrian',
    3: 'cyclist',
}


def get_label_map(mapping):
  """Get label id map based on the name, filename, or dict."""
  # case 1: if it is None or dict, just return it.
  if not mapping or isinstance(mapping, dict):
    return mapping

  if isinstance(mapping, hparams_config.Config):
    return mapping.as_dict()

  # case 2: if it is a yaml file, load it to a dict and return the dict.
  assert isinstance(mapping, str), 'mapping must be dict or str.'
  if mapping.endswith('.yaml'):
    with tf.io.gfile.GFile(mapping) as f:
      return yaml.load(f, Loader=yaml.FullLoader)

  # case 3: it is a name of a predefined dataset.
  return {'coco': coco, 'voc': voc, 'waymo': waymo}[mapping]

def parse_image_size(image_size: Union[Text, int, Tuple[int, int]]):
        """Parse the image size and return (height, width).

        Args:
            image_size: A integer, a tuple (H, W), or a string with HxW format.

        Returns:
            A tuple of integer (height, width).
        """
        if isinstance(image_size, int):
            # image_size is integer, with the same width and height.
            return (image_size, image_size)

        if isinstance(image_size, str):
            # image_size is a string with format WxH
            width, height = image_size.lower().split('x')
            return (int(height), int(width))

        if isinstance(image_size, tuple):
            return image_size

        raise ValueError('image_size must be an int, WxH string, or (height, width)'
                        'tuple. Was %r' % image_size)