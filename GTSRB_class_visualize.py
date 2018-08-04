import glob
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import os
import numpy as np

TRAIN_IMAGE_DIR = './data/GTSRB/Final_Training/Images'

dfs = []
for train_file in glob.glob(os.path.join(TRAIN_IMAGE_DIR, '*/GT-*.csv')):
    folder = train_file.split('/')[5]
    df = pd.read_csv(train_file, sep=';')
    df['Filename'] = df['Filename'].apply(lambda x: os.path.join(TRAIN_IMAGE_DIR, folder, x))
    dfs.append(df)

train_df = pd.concat(dfs, ignore_index=True)
print train_df.head()

N_CLASSES = np.unique(train_df['ClassId']).size  # keep this for later

print("Number of training images : {:>5}".format(train_df.shape[0]))
print("Number of classes         : {:>5}".format(N_CLASSES))

def show_class_distribution(classIDs, title):
    """
    Plot the traffic sign class distribution
    """
    plt.figure(figsize=(15, 5))
    plt.title('Class ID distribution for {}'.format(title))
    plt.hist(classIDs, bins=N_CLASSES)
    plt.ion()
    plt.pause(3)
    plt.close()

show_class_distribution(train_df['ClassId'], 'Train Data')

sign_name_df = pd.read_csv('sign_names.csv', index_col='ClassId')
sign_name_df['Occurence'] = [sum(train_df['ClassId'] == c) for c in range(N_CLASSES)]
sign_name_df.sort_values('Occurence', ascending=False)
SIGN_NAMES = sign_name_df.SignName.values

for i in range(0, len(SIGN_NAMES)):
    print "Class ", i+1, ":  ", SIGN_NAMES[i]

