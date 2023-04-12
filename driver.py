import numpy as np
import cv2
import time

import tensorflow as tf
from . import utils

colors = [0, (0, 255, 0), 0, (0, 128, 255)]

class EfficientdetDriver():
    def __init__(self):
        print("Init EfficientdetDriver")
        self.models = {}
        for backbone_idx in range(7):
            model_backbone = "d%d"%backbone_idx
            model_name = 'efficientdet-'+model_backbone
            self.models[model_name] = {}

    def load_model(self, model_name, saved_model_dir_or_frozen_graph, image_size='1920x1080'):
        #print(f"Build model {model_name} for {image_size}")
        
        def wrap_frozen_graph(graph_def, inputs, outputs):
            # https://www.tensorflow.org/guide/migrate
            imports_graph_def_fn = lambda: tf.import_graph_def(graph_def, name='')
            wrapped_import = tf.compat.v1.wrap_function(imports_graph_def_fn, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        graph_def = tf.Graph().as_graph_def()
        with tf.io.gfile.GFile(saved_model_dir_or_frozen_graph, 'rb') as f:
            graph_def.ParseFromString(f.read())

        model = wrap_frozen_graph(
            graph_def,
            inputs='images:0',
            outputs=['Identity:0', 'Identity_1:0', 'Identity_2:0', 'Identity_3:0'])
    
        #print(f"run warmup inference")
        _image_size = utils.parse_image_size(image_size)
        fake_inputs = np.ones(shape=(*_image_size, 3,))
        fake_inputs = tf.convert_to_tensor(np.array([fake_inputs]), dtype=tf.uint8)
        
        for i in range(10):
            start_time = time.time()
            _ = model(fake_inputs)
            inf_time = time.time() - start_time
            
        inf_times = []
        for i in range(10):
            start_time = time.time()
            _ = model(fake_inputs)
            inf_time = time.time() - start_time
            inf_times.append(inf_time)
        print(model_name, fake_inputs.shape, "inference time: %.4fs"%(np.mean(inf_times)))

        self.models[model_name][image_size] = model

    def inference(self, img, model_name, target_classes = None, image_size='1920x1080'):
        height = int(image_size.split("x")[1])
        width = int(image_size.split("x")[0])
        
        scale_h = img.shape[0]/height
        scale_w = img.shape[1]/width
        img = cv2.resize(img, (width,height))
        
        inputs = np.array([img])
        inputs = tf.convert_to_tensor(inputs, dtype=tf.uint8)

        if model_name in self.models.keys():
            model = self.models[model_name][image_size]
            outputs = model(inputs)
            boxes, scores, classes, inf_sizes = tf.nest.map_structure(np.array, outputs)
            
            boxes = self.scale_bboxes(boxes[0], scale_h, scale_w)
            scores = scores[0]
            classes = classes[0]
            inf_sizes = inf_sizes[0]
            
            if target_classes is not None:
                boxes, scores, classes = self.filter_output_by_classes(boxes, scores, classes, target_classes)

        return boxes, scores, classes, inf_sizes


    def visualize(self, image, boxes, scores, classes, target=None, score_thresh=0.4):
        for box, score, category in zip(boxes, scores, classes):
            if category == target and score > score_thresh:
                ymin, xmin, ymax, xmax = box
                ymin, xmin, ymax, xmax = int(ymin), int(xmin), int(ymax), int(xmax)
                label = f'{utils.coco[category]} {score:.2f}%'

                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=colors[target], thickness=2)
                cv2.rectangle(image, (xmin, ymin - h), (xmin + w, ymin), colors[target], -1)
                cv2.putText(image, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

        return image

    def filter_output_by_classes(self, boxes, scores, classes, target_classes):
        boxes_filtered = []
        scores_filtered = []
        classes_filtered = []
        
        for i in range(len(boxes)):
            #print(classes[i])
            if classes[i] in target_classes:
                boxes_filtered.append(boxes[i])
                scores_filtered.append(scores[i])
                classes_filtered.append(classes[i])
                
        return boxes_filtered, scores_filtered, classes_filtered
            
    def scale_bboxes(self, boxes, scale_h, scale_w):
        boxes_scaled = []
        for box in boxes:
            boxes_scaled.append([box[0]*scale_h, box[1]*scale_w, box[2]*scale_h, box[3]*scale_w])
        return boxes_scaled