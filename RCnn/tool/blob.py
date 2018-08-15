"""create imdb for training net  let origon imges for innput net sacle"""
import tensorflow as tf
import numpy as np
#import skimage
import cv2
#from  easydict import EasyDict as edit
"""
edit has a dic in blob
"""
def imglist_to_blob(img):
    """convert a list of imges to a net input"""
    max_shape=np.array([im.shape for im in img]).max(axis=0)  # get the max sacle
    num_imges=len(img)
    blob=np.zeros((num_imges,max_shape[0],max_shape[1],3),dtype=tf.float32)
    for i in range(num_imges):
        im=img[i]
        blob[i,0:max_shape[0],0:max_shape[1],:]=im
    return blob

def pre_imglist_for_blob(im,px_mean,target_size,max_size):
    """ prepare img for blob"""
    img=im.adtype(tf.float32,copy=False)
    im-=px_mean #减去平均像素
    im_shape=im.shape()
    im_size_min=np.min(im_shape[0:2])
    im_size_max=np.max(im_shape[0:2])
    im_sacle=(float)(target_size)/(float)(im_size_min)
    if np.round(im_sacle*im_size_max)>max_size:
        im_sacle=(float)(max_size)/(float)(im_size_max)
    im=cv2.resize(im,None,None,im_sacle,im_sacle,interpolation=cv2.INTER_LINEAR)
    return im,im_sacle
