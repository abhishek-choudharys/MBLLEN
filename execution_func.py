from glob import glob
import numpy as np
import scipy
import keras
import os
from MBLLEN.main.Network import * 
from MBLLEN.main.utls import *
import time
import cv2
import argparse


def enhance(input, output):

    argdict = dict()

    argdict["input"] = input
    argdict["result"] = output
    argdict["model"] = "Syn_img_lowlight_withnoise"
    argdict["com"] = 1
    argdict["highpercent"] = 95
    argdict["lowpercent"] =  5
    argdict["gamma"] = 8
    argdict["maxrange"] = 8 


    result_folder = argdict["result"]
    if not os.path.isdir(result_folder):
        os.makedirs(result_folder)

    input_folder = argdict["input"]
    path = glob(input_folder+'/*.*')

    model_name = argdict["model"]
    mbllen = build_mbllen((None, None, 3))
    mbllen.load_weights('MBLLEN/models/'+model_name+'.h5')
    opt = keras.optimizers.Adam(lr=2 * 1e-04, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mbllen.compile(loss='mse', optimizer=opt)

    flag = 0
    lowpercent = 5
    highpercent = 95
    maxrange = 8/10.
    hsvgamma = 8/10.

    img_A_path = input
    img_A = imread_color(img_A_path)
    img_A = img_A[np.newaxis, :]

    out_pred = mbllen.predict(img_A)
    print(out_pred.shape)

    image = np.squeeze(out_pred, axis=0)
    print(image.shape)

    fake_B = out_pred[0, :, :, :3]
    fake_B_o = fake_B

    gray_fake_B = fake_B[:, :, 0] * 0.299 + fake_B[:, :, 1] * 0.587 + fake_B[:, :, 1] * 0.114
    percent_max = sum(sum(gray_fake_B >= maxrange))/sum(sum(gray_fake_B <= 1.0))
    print(percent_max)


    max_value = np.percentile(gray_fake_B[:], highpercent)
    if percent_max < (100-highpercent)/100.:
        scale = maxrange / max_value
        fake_B = fake_B * scale
        fake_B = np.minimum(fake_B, 1.0)

    gray_fake_B = fake_B[:,:,0]*0.299 + fake_B[:,:,1]*0.587 + fake_B[:,:,1]*0.114
    sub_value = np.percentile(gray_fake_B[:], lowpercent)
    fake_B = (fake_B - sub_value)*(1./(1-sub_value))

    imgHSV = cv2.cvtColor(fake_B, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(imgHSV)
    S = np.power(S, hsvgamma)
    imgHSV = cv2.merge([H, S, V])
    fake_B = cv2.cvtColor(imgHSV, cv2.COLOR_HSV2RGB)
    fake_B = np.minimum(fake_B, 1.0)

    if flag:
        outputs = np.concatenate([img_A[0,:,:,:], fake_B_o, fake_B], axis=1)
    else:
        outputs = fake_B

    filename = os.path.basename(input)
    img_name = result_folder + '/' + filename
    # scipy.misc.toimage(outputs * 255, high=255, low=0, cmin=0, cmax=255).save(img_name)
    outputs = np.minimum(outputs, 1.0)
    outputs = np.maximum(outputs, 0.0)
    imwrite(img_name, outputs)
