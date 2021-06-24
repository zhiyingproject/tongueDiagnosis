import json
import glob
import cv2
import pandas as pd
from operator import itemgetter
from random import shuffle
import tables
import numpy as np


class preprocessing():
    def __init__(self, category, config_p, data_p, hdf5_p):
        self.category = category
        self.config_path = config_p
        self.data_path = data_p
        self.hdf5_path = hdf5_p

    def get_label_no(self):
        with open(f"{self.config_path}/labelName.json", "r") as f:
            label_mapping = json.load(f)[self.category]

        label_dict = [d for d in label_mapping if "NA" not in d["chineseName"]]
        chinese_labels = list(map(itemgetter('chineseName'), label_dict))
        numerical_labels = list(map(itemgetter('labelName'), label_dict))
        return_val = dict(zip(numerical_labels, chinese_labels))

        return return_val

    def clean_labels(self):
        labels_raw = pd.read_csv(f"{self.config_path}/labels.csv")
        label_no = self.get_label_no()
        print(label_no)
        lst_file = [f"{d}a.jpg" for d in labels_raw["fileName"]]
        if self.category != "top":
            lst_label = [list(label_no.keys())[list(label_no.values()).index(l_cn)]
                         for l_cn in labels_raw[self.category]]
        # shuffle data
        c = list(zip(lst_file, lst_label))
        shuffle(c)
        lst_file, lst_label = zip(*c)
        # return_val = dict(zip(lst_file, lst_label))
        return lst_file, lst_label

    def hdf5_generator(self):
        addrs, labels = self.clean_labels()

        train_addrs = addrs[0:int(0.6 * len(addrs))]
        train_labels = labels[0:int(0.6 * len(labels))]

        val_addrs = addrs[int(0.6 * len(addrs)):int(0.8 * len(addrs))]
        val_labels = labels[int(0.6 * len(labels)):int(0.8 * len(labels))]

        test_addrs = addrs[int(0.8 * len(addrs)):]
        test_labels = labels[int(0.8 * len(labels)):]

        print('train size:', len(train_addrs))
        print('val size:', len(val_addrs))
        print('test size:', len(test_addrs))

        img_dtype = tables.UInt8Atom()
        data_shape = (0, 128, 98, 3)
        hdf5_file = tables.open_file(self.hdf5_path, mode='w')

        try:
            train_storage = hdf5_file.create_earray(hdf5_file.root, 'train_img', img_dtype, shape=data_shape)
            val_storage = hdf5_file.create_earray(hdf5_file.root, 'val_img', img_dtype, shape=data_shape)
            test_storage = hdf5_file.create_earray(hdf5_file.root, 'test_img', img_dtype, shape=data_shape)
            mean_storage = hdf5_file.create_earray(hdf5_file.root, 'train_mean', img_dtype, shape=data_shape)
            hdf5_file.create_array(hdf5_file.root, 'train_labels', train_labels)
            hdf5_file.create_array(hdf5_file.root, 'val_labels', val_labels)
            hdf5_file.create_array(hdf5_file.root, 'test_labels', test_labels)
            mean = np.zeros(data_shape[1:], np.float32)
            for train_f in train_addrs:
                img = cv2.imread(f"{self.data_path}/{train_f}")
                img = cv2.resize(img, (98, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                train_storage.append(img[None])
                mean += img / float(len(train_labels))

            for val_f in val_addrs:
                img = cv2.imread(f"{self.data_path}/{val_f}")
                img = cv2.resize(img, (98, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                val_storage.append(img[None])
            f = open("./testfilename.txt", "w")
            for test_f in test_addrs:
                img = cv2.imread(f"{self.data_path}/{test_f}")
                img = cv2.resize(img, (98, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                test_storage.append(img[None])
                f.write(test_f + '\n')
            mean_storage.append(mean[None])
        finally:
            print("hdf5 is finished.")
            hdf5_file.close()




    def read_jpg(self):
        for i_f, f_name in enumerate(glob.glob(f"{self.data_path}/*.jpg")):
            print(f_name)


original_path = "../originalData/apparatus"
config_path = "../config"
hdf5_path = "../datasets/hdf5"

preprocessing_class = preprocessing("color", config_path, original_path, hdf5_path)
preprocessing_class.hdf5_generator()
