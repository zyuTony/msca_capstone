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

import siamesemodel

def siamese_eval_dataload(test_image, datafolder):
    prediction_df = []
    for label in os.listdir(datafolder):
        for image in os.listdir(datafolder + label):
            prediction_df.append({
              'test_img':test_image,
              'img2': datafolder + label + "/" + image,
              'labels': label,   
            })

    prediction_df = pd.DataFrame(prediction_df)
    prediction_df.to_csv("../data/prediction.csv", index=False)

    # load prediction csv
    prediction_csv = "../data/prediction.csv"

    prediction_dataset = siamesemodel.SiameseNetworkDataset(training_csv=prediction_csv,training_dir="",
                                          transform=transforms.Compose([transforms.Resize((105,105)),
                                                                        transforms.ToTensor()
                                                                        ]))

    prediction_dataloader = DataLoader(prediction_dataset, batch_size=1)

    return prediction_df, prediction_dataloader


def siamese_eval(test_image, datafolder, net, n=3):
    prediction_df, prediction_dataloader = siamese_eval_dataload(test_image, datafolder)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dissimilarity = []

    # Get dissimilarity for every combination of test img and our imgs
    for i, data in enumerate(prediction_dataloader,0): 
        test, compare, label = data
        concatenated = torch.cat((test, compare),0)
        output1, output2 = net(test.to(device), compare.to(device))
        eucledian_distance = F.pairwise_distance(output1, output2)
        dissimilarity.append(eucledian_distance.item()*10000)

    dissim_df = pd.DataFrame({"score":dissimilarity,
                            "labels":prediction_df['labels']})
  
    # Get avg dissimilarity score for each label
    labels = []
    scores = []
    rmsquares = []
    for label in set(prediction_df['labels']):
        scores_for_label = dissim_df[dissim_df['labels']==label]['score'].tolist()
        labels.append(label)
        scores.append(round(sum(scores_for_label)/len(scores_for_label), 2))
        rmsquare = np.sqrt(np.mean(np.square(scores_for_label)))
        rmsquares.append(rmsquare)

    # Get labels and the corresponding avg score
    avg_score_df = pd.DataFrame({"labels":labels,
                              "score":scores,
                              "root_mean_squared":rmsquares})
    
    top_n_pred = avg_score_df.sort_values("score", ascending=True)['labels'].head(n).tolist()
    top_n_pred_score = avg_score_df.sort_values("score", ascending=True)['score'].head(n).tolist()
   
    return top_n_pred, top_n_pred_score