import streamlit as st
import cv2, os, torch, pickle
from scipy import spatial

import torchvision
import torch.utils.data as utils
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import time
import copy
from torch.optim import lr_scheduler
import os
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd 

# import functions from other .py files
import imgs_functions
import predict
import siamesemodel

st.title('Dental Implant Identification')

@st.cache
def load_models():

    PATH = "./data/siamese_model.pt"
    net = siamesemodel.SiameseNetwork() 
    net.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    net.eval()

    yolo_presence = torch.hub.load('ultralytics/yolov5', 'custom', 'data/best-presence.pt')
    #yolo_type = torch.hub.load('ultralytics/yolov5', 'custom', '../../data/4-endtoend-models/best-type.pt') 
    yolo_presence.eval()
    #yolo_type.eval()
    #yolo_presence.cuda()
    #yolo_type.cuda()

    return net, yolo_presence

siamese_net, yolo_presence = load_models()

uploaded_file = st.file_uploader(
    label = 'Upload a Panoramix X-Ray containing an implant.', 
    type = ["png", "jpg", "jpeg"]
)

if uploaded_file:

    # show the image.
    img = Image.open(uploaded_file)
    img.save('./img/tempimage.jpg')

    img = np.array(img).astype('uint8')
    img_processed = imgs_functions.readandprocess(img)

    # Draw bounding box and display the image
    implants = yolo_presence('./img/tempimage.jpg').pandas().xyxy[0].sort_values('xmin')
    st.image(Image.fromarray(imgs_functions.drawBoundingBoxes(img, implants)))

    # crop and save implants as image
    for index, row in implants.iterrows():
        
        implantimg = img[
            int(row['ymin']):int(row['ymax']),
            int(row['xmin']):int(row['xmax'])
        ]
        Image.fromarray(implantimg).save('./img/'+str(index) +'.jpg')

    # Run prediction
    waiting = st.text('Running Siamese Network... Comparing test image to database...')

    datafolder = "../data/manual-clusters/20220430/categorized_new/"
    labels = []
    scores = []
    for i in range(len(implants)):
        test_img = './img/'+str(i) +'.jpg'
        label, score = predict.siamese_eval(test_img, datafolder, siamese_net, n=3)
        labels.append(label)
        scores.append(score)

    waiting.text('')
    for index, implant in implants.iterrows():

        st.image(Image.fromarray(img[
            int(implant['ymin']) : int(implant['ymax']),
            int(implant['xmin']) : int(implant['xmax'])
        ]))
        st.text("Top 3 Prediction for implant No." + str(index))
        for i in range(3):
            
            st.text(
                'Class:' + labels[index][i] + 
                ' Score: ' + str(scores[index][i])
                )
