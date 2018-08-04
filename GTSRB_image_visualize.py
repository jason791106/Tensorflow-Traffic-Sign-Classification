import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np

TRAIN_IMAGE_DIR = './data/GTSRB/Final_Training/Images'

def load_image(image_file):
    """
    Read image file into numpy array (RGB)
    """
    return plt.imread(image_file)


def get_samples(image_data, num_samples, class_id=None):
    """
    Randomly select image filenames and their class IDs
    """
    if class_id is not None:
        image_data = image_data[image_data['ClassId'] == class_id]
    indices = np.random.choice(image_data.shape[0], size=num_samples, replace=False)
    return image_data.iloc[indices][['Filename', 'ClassId']].values


def show_images(image_data, cols=5, sign_names=None, show_shape=False, func=None):
    """
    Given a list of image file paths, load images and show them.
    """
    num_images = len(image_data)
    rows = num_images//cols
    plt.figure(figsize=(cols*3,rows*2.5))
    for i, (image_file, label) in enumerate(image_data):
        # print image_file
        image = load_image(image_file)
        if func is not None:
            image = func(image)
        plt.subplot(rows, cols, i+1)
        plt.imshow(image)
        if sign_names is not None:
            plt.text(0, 0, '{}: {}'.format(label, sign_names[label]), color='k', backgroundcolor='c', fontsize=8)
        if show_shape:
            plt.text(0, image.shape[0], '{}'.format(image.shape), color='k', backgroundcolor='y', fontsize=8)
        plt.xticks([])
        plt.yticks([])
    plt.ion()
    plt.pause(3)
    plt.close()
dfs = []

for train_file in glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*/GT-*.csv')):
    folder = train_file.split('/')[5]
    df = pd.read_csv(train_file, sep=';')
    df['Filename'] = df['Filename'].apply(lambda x: os.path.join(TRAIN_IMAGE_DIR, folder, x))
    dfs.append(df)
train_df = pd.concat(dfs, ignore_index=True)
N_CLASSES = np.unique(train_df['ClassId']).size

sign_name_df = pd.read_csv('sign_names.csv', index_col='ClassId')
sign_name_df['Occurence'] = [sum(train_df['ClassId']==c) for c in range(N_CLASSES)]
sign_name_df.sort_values('Occurence', ascending=False)
SIGN_NAMES = sign_name_df.SignName.values

sample_data = get_samples(train_df, 20)
show_images(sample_data, sign_names=SIGN_NAMES, show_shape=True)

print(SIGN_NAMES[2])
show_images(get_samples(train_df, 100, class_id=2), cols=20, show_shape=True)



