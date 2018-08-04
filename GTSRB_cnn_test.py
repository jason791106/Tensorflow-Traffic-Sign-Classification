
import h5py
import numpy as np
import itertools
import cv2
from keras.models import Model
from keras.models import model_from_json
import matplotlib.pyplot as plt
import random
import ConfigParser
import pandas as pd
from sklearn.metrics import confusion_matrix

with h5py.File('./data/GTSRB_test.h5') as hf:
    X_test, Y_test = hf['imgs'][:], hf['labels'][:]
print "Loaded images from GTSRB_test.h5"

json_file = open('./models/GTSRB_classifier_architecture.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("./models/GTSRB_classifier_last_weights.h5")
model.summary()

y_pred = model.predict(X_test, verbose=1)
y_pred_class = np.argmax(y_pred, axis=1) + 1

test_result = np.zeros(y_pred_class.shape)
test_result[y_pred_class == Y_test] = 1

acc = np.sum(test_result)/np.size(y_pred_class)
print("Test accuracy = {}".format(acc))




def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 fontsize=5,
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cnf_matrix = confusion_matrix(Y_test, y_pred_class)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
class_names = range(1, 44, 1)
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
