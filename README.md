# Tensorflow Traffic Sign Classification

tensorflow traffic sign classification

## Step1. Train and Test Data

I build this project using the original image dataset from **German Traffic Sign Benchmarks** website.

<br />

### Download link
***
- [GTSRB_Final_Training_Images.zip](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Training_Images.zip) 
-  [GTSRB_Final_Test_Images.zip](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_Images.zip) 
- [GTSRB_Final_Test_GT.zip](http://benchmark.ini.rub.de/Dataset/GTSRB_Final_Test_GT.zip) 

<br />

### Folder Structure
***
```
data
  + GTSRB
    + Final_Training
      + Images
    + Final_Test
      + Images
    + GT-final_test.csv
```

<br />

##  Step2. Class Visualize and Image Visualize

You could get total class number and traffic sigh names for each class by running **GTSRB_class_visualize.py**

```
python GTSRB_class_visualize.py
```

If you want to preview some of the images form training data, please run **GTSRB_image_visualize.py**

```
python GTSRB_image_visualize.py
```

<br />

## Step3. Create Training and Testing h5 Datasets

You should create **GTSRB_test.h5 **and **GTSRB_train.h5** first before you start training.

```
python GTSRB_build_h5_dataset.py
```

<br />

## Step4. Training CNN Model

```
python GTSRB_cnn_train.py
```

<br />

## Step5. Testing CNN Model

```
python GTSRB_cnn_test.py 
```