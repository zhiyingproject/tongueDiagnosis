import sys
import matplotlib.pylab as plt
from collections import Counter
import pandas as pd

sys.path.insert(0, './myClass/')
import preprocessing
import create_model


original_path = "../originalData/apparatus"
config_path = "../config"
subtract_mean = True
epoch = 16
img_height = 166
img_width = 128
input_shape = (img_height, img_width, 3)
categories = ['spike', 'color', 'shape', 'crack', 'coatingColor', 'coatingAmount']

for category in categories:
    hdf5_path = f"../datasets/hdf5{category}"
    preprocessing_class = preprocessing.hdf5File(category, config_path, original_path, hdf5_path,
                                                 img_width, img_height)
    files, labels = preprocessing_class.clean_labels()
    label_map = preprocessing_class.get_label_no()
    print(label_map)
    label_count = Counter(labels)
    names = label_count.keys()
    values = label_count.values()
    plt.bar(names,values)
    plt.show()
