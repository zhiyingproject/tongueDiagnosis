import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from keras.utils import np_utils
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import sys
sys.path.insert(0, './myClass/')
import preprocessing
from collections import Counter
import pandas as pd
import pathlib
import tensorflow_addons as tfa
import cv2
import mahotas
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import cohen_kappa_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
import h5py
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.datasets import ad_hoc_data
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score



ds_p = "../datasets"
config_f = '../config/labels.csv'
scoring = "cohen_kappa_score"

labels = pd.read_csv(config_f)
categories = labels.columns[1:]


batch_size = 32
img_height = 169
img_width = 128
bins = 8
seed = 1234
num_trees = 100
algorithm_globals.random_seed = seed

models = []
models.append(('SVM', SVC(random_state=seed)))
models.append(("RF", RandomForestClassifier(n_estimators=num_trees)))
# preprocessing.organize_f(original_p, to_p, config_f)

def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def fd_haralick(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(gray.astype(int)).mean(axis=0)
    return haralick

def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()



adhoc_dimension = 532
'''train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False, one_hot=False, include_sample_total=True
)
print(train_features.shape, train_labels.shape)
'''


for category in categories[2:3]:
    print(category)
    data_dir = f'{ds_p}/{category}'
    data_dir = pathlib.Path(data_dir)
    h5_data_path = f'{str(data_dir)}/data.h5'
    h5_label_path = f'{str(data_dir)}/label.h5'
    image_count = len(list(data_dir.glob('*/*.jpg')))
    label_names = [p.name for p in data_dir.rglob('*') if p.is_dir()]
    num_classes = len(label_names)
    labels = []
    global_features = []
    for l_name in label_names:
        img_dir = pathlib.Path(f'{str(data_dir)}/{l_name}')
        for img in list(pathlib.Path(img_dir).glob("*")):
            labels.append(l_name)
            image = cv2.imread(str(img))
            image = cv2.resize(image, (img_height, img_width))
            fv_hu_moments = fd_hu_moments(image)
            fv_haralick = fd_haralick(image)
            fv_histogram = fd_histogram(image)
            global_feature = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
            global_features.append(global_feature)

    label_numerical = LabelEncoder().fit_transform(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize The feature vectors...
    rescaled_features = scaler.fit_transform(global_features)

    h5f_data = h5py.File(h5_data_path,'w')
    h5f_data.create_dataset('ds1', data=np.array(rescaled_features))

    h5f_label = h5py.File(h5_label_path, 'w')
    h5f_label.create_dataset('ds1', data=np.array(label_numerical))

    (train_ds, test_ds, train_labels, test_labels) = train_test_split(
        np.array(rescaled_features),
        np.array(label_numerical),
        test_size=0.1,
        random_state=seed
    )
    # !!!!!!Quantum

    feature_map = ZZFeatureMap(feature_dimension=dimensions,
                                     reps=2, entanglement='linear')

    backend = QuantumInstance(BasicAer.get_backend('qasm_simulator'), shots=1024,
                                    seed_simulator=seed, seed_transpiler=seed)

    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=backend)

    adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
    print("start to fit, using the model evaluate...")
    adhoc_svc.fit(train_ds, train_labels)
    adhoc_score = adhoc_svc.score(test_ds, test_labels)

    print(f'Callable kernel classification test score: {adhoc_score}')

    adhoc_matrix_train = adhoc_kernel.evaluate(x_vec=train_features)
    adhoc_matrix_test = adhoc_kernel.evaluate(x_vec=test_features,
                                              y_vec=train_features)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.asmatrix(adhoc_matrix_train),
                  interpolation='nearest', origin='upper', cmap='Blues')
    axs[0].set_title("Ad hoc training kernel matrix")
    axs[1].imshow(np.asmatrix(adhoc_matrix_test),
                  interpolation='nearest', origin='upper', cmap='Reds')
    axs[1].set_title("Ad hoc testing kernel matrix")
    plt.show()

    adhoc_svc = SVC(kernel='precomputed')
    adhoc_svc.fit(adhoc_matrix_train, train_labels)
    adhoc_score = adhoc_svc.score(adhoc_matrix_test, test_labels)

    print(f'Precomputed kernel classification test score: {adhoc_score}')

    qsvc = QSVC(quantum_kernel=adhoc_kernel)
    qsvc.fit(train_ds, train_labels)
    qsvc_score = qsvc.score(test_ds, test_labels)

    print(f'QSVC classification test score: {adhoc_score}')



    sys.exit(0)
    names = []
    results = []
    for name, model in models:
        kfold = KFold(n_splits=10)
        cv_results = cross_val_score(model, train_ds, train_labels, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # boxplot algorithm comparison
    fig = plt.figure()
    fig.suptitle('Machine Learning algorithm comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()