# efficientdetLite
Simple driver to use frozen efficientdet model


# How to use
1. Import the directory  
```command
git clone https://github.com/jiku100/efficientdetLite.git
```

2. Demo  

**Directory structure** 
```bash
root
|-- efficientdetLite
|   |-- driver.py
|   |-- hparams_config.py
|   |-- utils.py
|
|-- infer.py
```
**infer.py**  
```python  
from efficientdetLite.driver import EfficientdetDriver
from efficientdetLite.utils import *
import cv2

model_name = "efficientdet-d0"
model_path = "1080p/efficientdet-d0_frozen.pb"
image_size = "1920x1080"

## GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

## Declare driver
driver = EfficientdetDriver()

## Load model
driver.load_model(model_name, model_path, image_size)

image = cv2.imread("img1.jpg")

## Inference
boxes, scores, classes, _ = driver.inference(image, model_name, image_size=image_size)

## Visualize

target_index = 3 ## 1: person, 3: car
score_thresh = 0.4 ## Confidence threshold

result = driver.visualize(image, boxes, scores, classes, target_index, score_thresh)

cv2.imwrite("output.jpg", result)

```