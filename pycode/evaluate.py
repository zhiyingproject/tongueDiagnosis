from mrcnn.utils import Dataset, extract_bboxes, compute_ap
from mrcnn.config import Config
from mrcnn.visualize import display_instances
from numpy import zeros, asarray, expand_dims, mean
from mrcnn.model import MaskRCNN, load_image_gt, mold_image
import pathlib
import xml.etree.ElementTree as ET
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

'''
    scikit-image == 0.16.2
    tensorflow == 1.15.0
    keras == 2.2.4
'''
training_threshold = 140


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


class predictionConfig(Config):
    NAME = 'tongue_cfg'
    NUM_CLASSES = 1 + 1
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def evaluate_model(ds, model, cfg):
    APs = list()
    for i, img_id in enumerate(ds.image_ids):
        print(ds.image_info[i])
        image, image_meta, gt_class_id, gt_bbox, gt_mask = load_image_gt(ds, cfg, img_id, use_mini_mask=False)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)
        r = yhat[0]
        AP, _, _, _ = compute_ap(gt_bbox, gt_class_id, gt_mask, r['rois'], r['class_ids'], r['scores'], r['masks'])
        print(AP)
        if math.isnan(AP):
            continue
        APs.append(AP)
    mAP = mean(APs)
    return mAP


def plot_actual_vs_predicted(ds, model, cfg, n_image=2):
    for i in range(n_image):
        image = ds.load_image(i)
        mask, _ = ds.load_mask(i)
        scaled_image = mold_image(image, cfg)
        sample = expand_dims(scaled_image, 0)
        yhat = model.detect(sample, verbose=0)[0]
        plt.subplot(n_image, 2, i*2+1)
        plt.imshow(image)
        plt.title('Actual')
        for j in range(mask.shape[2]):
            plt.imshow(mask[:, :, j], cmap='gray', alpha=0.3)
        plt.subplot(n_image, 2, i*2+2)
        plt.imshow(image)
        plt.title('Predict')
        ax = plt.gca()
        for box in yhat['rois']:
            y1, x1, y2, x2 = box
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, fill=False, color='red')
            ax.add_patch(rect)
    plt.show()





img_path = "../segTraining"

training_ds = tongueDataset()
training_ds.load_dataset(img_path, is_train=True)
training_ds.prepare()
print('Train: %d' % len(training_ds.image_ids))

test_ds = tongueDataset()
test_ds.load_dataset(img_path, is_train=False)
test_ds.prepare()
print('Test: %d' % len(test_ds.image_ids))

img_test_path = "../segTest"
test_ds_outside = tongueDataset()
test_ds_outside.load_dataset(img_test_path, is_train=True)
test_ds_outside.prepare()

weights_file = 'mask_rcnn_tongue_cfg_0005.h5'  # !!!!!need to identify
cfg = predictionConfig()
model = MaskRCNN(mode='inference', model_dir='./', config=cfg)
model.load_weights(weights_file, by_name=True)

plot_actual_vs_predicted(test_ds_outside, model, cfg)

sys.exit(0)

train_mAP = evaluate_model(training_ds, model, cfg)
print("Train mAP: %.3f" % train_mAP)
# evaluate model on test dataset
# test_mAP = evaluate_model(test_ds, model, cfg)
# print("Test mAP: %.3f" % test_mAP)
