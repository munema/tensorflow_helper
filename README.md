# tensorflow_helper
my tensorflow helper code

## get_frozen_model.py
- Get keras pretrained model (.pb file)
- You can choose pretrained model at https://keras.io/ja/applications/
- you can use pretrained model at tensorflow.

### Requirement
Tensorflow version 1.15
```python
import tensorflow as tf
print(tf.version)
# 1.15
```

### Useage
```python
python get_frozen_model.py dir/output_file_name.pb
```

### Example
```python
python get_frozen_model.py tmp/frozen_model.pb
```
