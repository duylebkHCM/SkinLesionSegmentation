# Skin Lesion Segmentation
In this project I reimplement Unet model from srcatch base on Unet paper using Tensorflow and perform segmentation on ISIC 2018 dataset which a skin lesion dataset include 2594 training images. After that I create a app to serve the model using FastAPI as backend and Streamlit as frontend and combine them using Docker.

## DATASET
For this project I will use ISIC 2018 dataset which can be downloaded here [Link Download](https://challenge.isic-archive.com/data#2018)
This dataset include 2594 image-mask pair for segmenting lesion on skin.

<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/download.png" alt="isic" width=600 />

## UNET MODEL
Unet model contain 2 part: Encoder and Decoder. 
* Encoder: Use to extract features from the image, from low level features to high level features. It include 5 block, each block is a group of convolution, batchnorm and max-pooling layer
* Decoder: High level features will go through the decoder to recover spatial information, each step the size of features will get double, also to prevent from loss spatial information it include the features from encoder through what call skip-connection.

<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/unet.png" alt="unet" width=600 />

## RESULT
I train the model for 60 epoch using Adam optimizer, combine loss between dice loss and binary cross entropy loss. The model achieves best dice coefficience of 81.49 

<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/plot.png" alt="loss" width=600 />

### Some predicted images:
<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/predict.png" alt="pred" width=600 />

## WEB APP
I serve an Unet model for skin lesion segmentation using `FastAPI` for the backend service and `streamlit` for the frontend service. `docker-compose` orchestrates the two services and allows communication between them.

To run the example in a machine running Docker and docker-compose, run:
    docker-compose build
    docker-compose up

To visit the FastAPI documentation of the resulting service, visit http://localhost:8000 with a web browser.  
To visit the streamlit UI, visit http://localhost:8501.

### Some images from the app

<p align="center">
<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/exam1.png" alt="x1" width=600 />
<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/exam2.png" alt="x2" width=600 />
<img src="https://github.com/duylebkHCM/SkinLesionSegmentation/blob/master/assets/exam3.png" alt="x3" width=600 />
</p>

## REFERENCE
O.  Ronneberger,  P.  Fischer,  and  T.  Brox,  “U-net:  Con-volutional  networks  for  biomedical  image  segmentation,”vol. abs/1505.04597, 2015.
