# Script to make hdf5 files from training and test set
import numpy as np
from skimage import io, color, exposure, transform
import pandas as pd
import os
import glob
import h5py


NUM_CLASSES = 43
IMG_SIZE = 48

def preprocess_img(img):
    # Histogram normalization in y
    hsv = color.rgb2hsv(img)
    hsv[:,:,2] = exposure.equalize_hist(hsv[:,:,2])
    img = color.hsv2rgb(hsv)

    # central scrop
    min_side = min(img.shape[:-1])
    centre = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]

    # rescale to standard size
    img = transform.resize(img, (IMG_SIZE, IMG_SIZE))

    return img


def get_class(img_path):
    return int(img_path.split('/')[-2])

if __name__ == '__main__':

    #'''

    root_dir = './data/GTSRB/Final_Training/Images/'
    imgs = []
    labels = []
    X_train = []
    y_train = []
    all_train_img_paths = glob.glob(os.path.join(root_dir, '*/*.ppm'))
    np.random.shuffle(all_train_img_paths)
    for img_path in all_train_img_paths:
        try:
            img = preprocess_img(io.imread(img_path))
            label = get_class(img_path)
            imgs.append(img)
            labels.append(label)

            if len(imgs)%1000 == 0: print("Train Data Processed {}/{}".format(len(imgs), len(all_train_img_paths)))
        except (IOError, OSError):
            print('missed', img_path)
            pass

    X_train = np.array(imgs, dtype='float32')
    Y_train = np.eye(NUM_CLASSES, dtype='uint8')[labels]

    with h5py.File('./data/GTSRB_train.h5','w') as hf:
        hf.create_dataset('imgs', data=X_train)
        hf.create_dataset('labels', data=Y_train)

    #'''

    test = pd.read_csv('./data/GTSRB/GT-final_test.csv',sep=';')

    X_test = []
    y_test = []
    i = 0
    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
        img_path = os.path.join('./data/GTSRB/Final_Test/Images/', file_name)
        X_test.append(preprocess_img(io.imread(img_path)))
        y_test.append(class_id+1)
        if len(X_test) % 1000 == 0: print("Test Data Processed {}/{}".format(len(X_test), len(list(test['Filename']))))

    X_test = np.array(X_test, dtype='float32')
    y_test = np.array(y_test, dtype='uint8')

    with h5py.File('./data/GTSRB_test.h5','w') as hf:
        hf.create_dataset('imgs', data=X_test)
        hf.create_dataset('labels', data=y_test)
