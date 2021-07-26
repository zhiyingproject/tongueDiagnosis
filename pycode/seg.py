import pandas as pd
import pathlib
import shutil
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import sys

ds_p = pathlib.Path("../datasets")
to_p = "../segTraining"
img_height = 169
img_width = 128
all_files = ds_p.rglob('*.jpg')


def re_size():
    for f in all_files:
        img = cv2.imread(str(f))
        img = cv2.resize(img,(img_width, img_height))
        cv2.imwrite(f'{to_p}/{str(f.name)}',img)


for f in list(all_files)[10:20]:
    annot_f = f'{to_p}/annots/{f.stem}.xml'
    from_f = pathlib.Path(f"{to_p}/annots/template.xml")
    shutil.copy(from_f, annot_f)
    tree = ET.parse(annot_f)
    root = tree.getroot()
    root.find('filename').text = f'{f.stem}.jpg'
    img = cv2.imread(f'{to_p}/{f.name}')
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



'''txt = input("xmin")
print(txt)'''