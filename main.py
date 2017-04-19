import os
from detector import Detector

head = 'example'
filename_cfg = os.path.join('{}.cfg'.format(head))
weights = '{}.hdf5'.format(head)
filename_weights = os.path.join('weights',weights)

## Forward pass
detector = Detector(filename_cfg,GPU=True)
detector.load_framework(framework)
layer_output = detector.eval_layer(layer,img_input)