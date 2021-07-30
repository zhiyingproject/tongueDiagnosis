import pandas as pd
import pathlib
import shutil
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys
from mrcnn.utils import Dataset, extract_bboxes, compute_ap
from mrcnn.config import Config
from mrcnn.visualize import display_instances
from numpy import zeros, asarray, expand_dims, mean
from mrcnn.model import MaskRCNN, load_image_gt, mold_image

'''
    scikit-image == 0.16.2
    tensorflow == 1.15.0
    keras == 2.2.4
'''


ds_p = pathlib.Path("../originalDS")
to_p = "../segTraining"
img_height = 169
img_width = 128
all_files = list(pathlib.Path(to_p).rglob('*.jpg'))
training_threshold = 140

def re_size(ds_dir):
    img_files = list(pathlib.Path(ds_dir).rglob('*.jpg'))
    for f in img_files:
        img = cv2.imread(str(f))
        img = cv2.resize(img, (img_width, img_height))
        cv2.imwrite(f'{ds_dir}/{str(f.name)}', img)
    return


def annotation(ds_dir):
    img_files = list(pathlib.Path(ds_dir).rglob('*.jpg'))
    for f in img_files:  # !new portion
        annot_f = f'{ds_dir}/annots/{f.stem}.xml'
        if pathlib.Path(annot_f).is_file():
            print(annot_f)
            continue
        print(annot_f)
        from_f = pathlib.Path(f"{ds_dir}/annots/template.xml")
        shutil.copy(from_f, annot_f)
        tree = ET.parse(annot_f)
        root = tree.getroot()
        root.find('filename').text = f'{f.stem}.jpg'
        img = cv2.imread(f'{ds_dir}/{f.name}')
        plt.imshow(img)
        plt.show()
        xmin = input("xmin")
        xmax = input("xmax")
        ymin = input("ymin")
        ymax = input("ymax")

        for box in root.findall('.//bndbox'):
            box.find('xmin').text = xmin
            box.find('xmax').text = xmax
            box.find('ymin').text = ymin
            box.find('ymax').text = ymax

        tree.write(annot_f)


def extract_boxes1(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    boxes = list()
    for box in root.findall('.//bndbox'):
        xmin = int(box.find('xmin').text)
        ymin = int(box.find('ymin').text)
        xmax = int(box.find('xmax').text)
        ymax = int(box.find('ymax').text)
        coors = [xmin, ymin, xmax, ymax]
        boxes.append(coors)
    width = int(root.find('.//size/width').text)
    height = int(root.find('.//size/height').text)
    return boxes, width, height


class tongueDataset(Dataset):
    def load_dataset(self, dataset_dir, is_train=True):
        self.add_class("dataset", 1, "tongue")

        images_dir = dataset_dir
        annotations_dir = f'{dataset_dir}/annots'
        image_files = pathlib.Path(dataset_dir).rglob('*.jpg')
        for ifile, filename in enumerate(list(image_files)):
            image_id = filename.stem
            if is_train and ifile >= training_threshold:
                continue
            if not is_train and ifile < training_threshold:
                continue
            img_path = filename
            ann_path = f'{annotations_dir}/{filename.stem}.xml'
            self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

    def extract_boxes(self, filename):
        tree = ET.parse(filename)
        root = tree.getroot()
        boxes = list()
        for box in root.findall('.//bndbox'):
            xmin = int(box.find('xmin').text)
            ymin = int(box.find('ymin').text)
            xmax = int(box.find('xmax').text)
            ymax = int(box.find('ymax').text)
            coors = [xmin, ymin, xmax, ymax]
            boxes.append(coors)
        width = int(root.find('.//size/width').text)
        height = int(root.find('.//size/height').text)
        return boxes, width, height

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info['annotation']
        boxes, w, h = self.extract_boxes(path)
        masks = zeros([h, w, len(boxes)], dtype='uint8')

        class_ids = []
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('tongue'))
        return masks, asarray(class_ids, dtype='int32')

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class tongueConfig(Config):
    NAME = 'tongue_cfg'
    NUM_CLASSES = 1 + 1
    STEPS_PER_EPOCH = training_threshold



re_size('../segTest')
annotation("../segTest")
sys.exit(0)




training_ds = tongueDataset()
training_ds.load_dataset(to_p, is_train=True)
training_ds.prepare()
print('Train: %d' % len(training_ds.image_ids))

test_ds = tongueDataset()
test_ds.load_dataset(to_p, is_train=False)
test_ds.prepare()
print('Test: %d' % len(test_ds.image_ids))

'''image_id = 1
image = training_ds.load_image(image_id)
mask, class_ids = training_ds.load_mask(image_id)
bbox = extract_bboxes(mask)
display_instances(image, bbox, mask, class_ids, training_ds.class_names)'''

'''for image_id in training_ds.image_ids:
    image = training_ds.load_image(image_id)
    info = training_ds.image_info[image_id]
    mask, class_ids = training_ds.load_mask(image_id)
    print(info)'''

'''plt.imshow(image)
   plt.imshow(mask[:, :, 0], cmap='gray', alpha=0.5)
   plt.show()'''

config = tongueConfig()
config.display()

model = MaskRCNN(mode='training', model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
model.train(training_ds, test_ds, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')
