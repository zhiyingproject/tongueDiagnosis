import h5py
import pandas as pd
import pathlib
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

config_f = '../config/labels.csv'

labels = pd.read_csv(config_f)
categories = labels.columns[1:]

p_ds = '../datasets'
p_ds_cropped = '../dsCropped'
img_height = 169
img_width = 128

for category in categories[:1]:
    data_dir = f'{p_ds}/{category}'
    data_dir = pathlib.Path(data_dir)
    label_names = [d.name for d in data_dir.rglob('*') if d.is_dir()]
    num_classes = len(label_names)
    for label in label_names:
        img_dir = f'{str(data_dir)}/{label}'
        for f_img in list(pathlib.Path(img_dir).glob('*'))[:2]:
            image = cv2.imread(str(f_img))
            image = cv2.resize(image, (img_width, img_height))
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gp = cv2.GaussianBlur(image_gray,(5,5),cv2.BORDER_DEFAULT)

            '''# contour method
            ret, thresh = cv2.threshold(image_gray, 120, 255, cv2.THRESH_BINARY)
            # Now finding Contours         ###################
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            coordinates = []
            image_copy = image.copy()
            for cnt in contours:
                # [point_x, point_y, width, height] = cv2.boundingRect(cnt)
                approx = cv2.approxPolyDP(
                    cnt, 0.07 * cv2.arcLength(cnt, True), True)
                if len(approx) == 3:
                    coordinates.append([cnt])
                    cv2.drawContours(image_copy, [cnt], 0, (0, 0, 255), 3)

            cv2.imshow('binary image', image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

            '''# using edge detection
            # edges = cv2.Canny(image_gray, 30, 80)
            # edges = cv2.Laplacian(image_gray, cv2.CV_64F)

            sobel_horizontal = np.array([np.array([1, 2, 1]), np.array([0, 0, 0]), np.array([-1, -2, -1])])
            sobel_vertical = np.array([np.array([-1, 0, 1]), np.array([-2, 0, 2]), np.array([-1, 0, 1])])

            cv2.imshow("test",np.hstack((image_gray, edges)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()'''

            # using clustering
            image_n = image.reshape(image.shape[0]*image.shape[1], image.shape[2])
            k_model = KMeans(n_clusters=3, random_state=0).fit(image_n)
            image_show = k_model.cluster_centers_[k_model.labels_]
            print(k_model.labels_)
            cluster_image = image_show.reshape(image.shape[0],image.shape[1],image.shape[2])
            plt.imshow(cluster_image)
            plt.show()
            continue
            image_copy = image.copy()
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1,
                             color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)

            # cropped = image_copy[y:y+h, x:x+w]
            cv2.imshow('binary image', image_copy)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
