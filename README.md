## Pneumonia Detection using Chest X-Ray
Dataset: https://www.kaggle.com/datasets/umitka/chest-x-ray-balanced

This dataset is an augmented and partitioned version of paultimothymooney's chest-xray-pneumonia dataset, with the images divided into 10% test, 10% validation, and 80% train folders. These steps were taken to create a more balanced dataset. In its augmented form, the test folder contains 400 PNEUMONIA and 400 NORMAL images; the validation folder contains 400 PNEUMONIA and 400 NORMAL images; and the training folder contains 4000 PNEUMONIA and 4000 NORMAL images.

Overview

This repository contains the Pneumonia Detection from Chest X-ray Images project, which uses deep learning to classify chest X-ray images as Normal or Pneumonia. Chest X-ray imaging is a common diagnostic tool for pneumonia, but manual interpretation is time-consuming and depends on radiologist availability. This project builds and compares multiple classification approaches, then deploys the best-performing model as a locally-runnable API that accepts an X-ray image and returns a prediction.

Project Workflow

1. Exploratory Data Analysis
Performed on the full 9,600-image dataset, covering data loading, cleaning (corruption, duplicate, and file-size checks), class distribution, sample image inspection, image dimension and color mode analysis, pixel intensity comparison between classes, and average-image comparison. Findings directly informed the preprocessing pipeline used across all models.

2. Preprocessing and Feature Engineering

All images are resized to 224×224, converted to RGB, and normalized to a 0–1 pixel range. Data augmentation (rotation, width/height shift, zoom, brightness adjustment) is applied to the training set only, with horizontal flipping deliberately excluded since mirroring an X-ray does not reflect a realistic clinical image.

3. Model Development

Three distinct classification approaches were implemented and evaluated on the same held-out test set:

Model Type Description

CNN: Built from scratch	Three Conv2D/MaxPooling blocks with a dense classification head, used as the baseline.

DenseNet121:	Transfer learning	Pretrained on ImageNet, frozen base with a custom classification head, following the architecture used in CheXNet.

Random Forest:	Hybrid ML	Trained on feature vectors extracted from the frozen DenseNet121 base, rather than raw pixels.

4. Evaluation

Each model was assessed on accuracy, precision, recall, F1-score, confusion matrix, and ROC-AUC curve, using the untouched test set. Results showed the CNN achieving the strongest overall accuracy and precision, DenseNet121 and Random Forest achieving higher recall (fewer missed Pneumonia cases), and Random Forest achieving the highest AUC — reflecting a genuine trade-off between minimizing missed diagnoses and minimizing false alarms rather than one model being universally best.

5. Deployment

The trained CNN model is deployed locally as a REST API using FastAPI and uvicorn, allowing a chest X-ray image to be uploaded and classified as Normal or Pneumonia with a confidence score.