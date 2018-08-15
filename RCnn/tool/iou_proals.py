import numpy as np
import pickle
from PIL import Image
#import os.path
import tensorflow as tf
import selectivesearch
import skimage
'''
计算候选区域以及iou
'''
def pil_nparr(pil):
    return np.array(pil.load(),dtype=tf.float32)

def img_resize(i_img,new_width,new_height,out_img=None,rresize_modle=Image.ANTIALIAS):
    img=i_img.resize((new_width,new_height),rresize_modle)
    if out_img:
        img.save(out_img)
    return  img

def if_interection(xmin_a,xmax_a,ymin_a,ymax_a,xmin_b,xmax_b,ymin_b,ymax_b):
    '''
    计算两个候选区域是否交叉
    :param xmin_a:
    :param xmax_a:
    :param ymin_a:
    :param ymax_a:
    :param xmin_b:
    :param xmax_b:
    :param ymin_b:
    :param ymax_b:
    :return: area_inter
    '''
    if_inserect=False
    if xmin_a<xmax_b<=xmax_a and (ymin_a<ymax_b<=ymax_a or ymin_a<=ymax_b<ymax_a):
        if_inserect=True
    elif xmin_a<=xmin_b<xmax_a and (ymin_a<ymax_b<=ymax_a or ymin_a<=ymax_b<ymax_a):
        if_inserect=True
    elif xmin_b<xmax_a<=xmax_b and (ymin_b<ymax_a<=ymax_b or ymin_b<=ymax_a<ymax_b):
        if_inserect=True
    elif xmin_b<=xmax_a<xmax_b and (ymin_b<ymax_a<=ymax_b or ymin_b<=ymax_a<ymax_b):
        if_inserect=True
    else:
        return  False
    if if_inserect:
        x_sorted_list=sorted([xmin_a,xmax_a,xmin_b,xmax_b])
        y_sorted_list=sorted([ymin_b,ymax_b,ymin_a,ymax_a])
        x_intersect_w=x_sorted_list[2]-x_sorted_list[1]
        y_intersect_h=y_sorted_list[2]-y_sorted_list[1]
        area_iner=x_intersect_w*y_intersect_h
        return  area_iner

    def calcuate_iou(ver1,ver2):
        '''
        根据传入的四个点计算两个举行框的iou
        :param ver1:
        :param ver2:
        :return: float iou
        '''
        vertercel1=[ver1[0],ver1[1],ver1[0]+ver1[2],ver1[1]+ver1[3]]
        verterce2=[ver2[0],ver2[1],ver2[0]+ver2[2],ver2[1]+ver2[3]]
        areas=if_interection(vertercel1[0],vertercel1[2],vertercel1[1],vertercel1[3],verterce2[0],verterce2[2],verterce2[1],verterce2[3])
        if areas:
            area1=vertercel1[2]*vertercel1[3]
            area2=verterce2[2]*verterce2[3]
            iou_float=tf.float32(areas/(area1+area2-areas))
            return  iou_float
        return  False

    def clip_pil(img,rect):
        '''
        :param img:
        :param rect:
        :return: the rect of pil list of w and h
        '''
        x=rect[0]
        y=rect[1]
        w=rect[2]
        h=rect[3]
        y_to=rect[3]+rect[1]
        x_to=rect[2]+rect[0]
        return img[x:x_to,y:y_to],[x,y,w,h]

    def load_train_proposals(datapath,num_class,threshold = 0.5, svm = False, save=False, save_path='dataset.pkl'):
        '''
        筛选iou符合阈值的边框里图片数据以及图片对应的标签
        :param datapath:
        :param num_class:
        :param threshold:
        :param svm:
        :param save:
        :param save_path:
        :return:clip_img,rect
        '''
        train_list=open(datapath,"rb")
        images=[]
        labels=[]
        for temp in train_list:
            line=temp.rstrip().split(' ')
            img=skimage.io.imread(line[0])
            img_label,img_reigons=selectivesearch.selective_search(img,scale=500,sigma=0.9,min_size=10)
            candidates =set()
            print(line[0])
            for r in img_reigons:
                if r['rect'] in candidates:
                    continue
                if r['size']<220:
                    continue
                porposal_img,porposal_vertecl=clip_pil(img,r['rect'])
                if len(porposal_img)==0:
                    continue
                x,y,w,h=r['rect']
                if w==0 or h==0:
                    continue
                a,b,c=np.shape(porposal_img)
                if a==0 or b==0 or c==0:
                    continue
                # resize pil to *222244
                reserial_img=Image.fromarray(porposal_img)
                img_re=img_resize(reserial_img,224,224)
                # 记录rect
                candidates.add(r['rect'])
                floa_img=pil_nparr(img_re)
                # 将图片数据加入images
                images.append(floa_img)
                # iou
                ref_iou=line[2].split(',')
                ref_ioucon=[int(i) for i in ref_iou]
                iou_val=calcuate_iou(ref_ioucon,porposal_vertecl)
                index=int(line[1])
                if svm==False:
                    label=np.zeros(num_class+1)
                    if iou_val<threshold:
                        label[0]=1
                    else:
                        label[index]=1
                    labels.append(label)
                else:
                    if iou_val<threshold:
                        labels.append(0)
                    else:
                        labels.append(index)
                if save:
                    pickle.dump((images,labels),open(save_path,'wb'))
                return images,labels


    def load_from_ppick(path):
        x,y=pickle.load(path,'rb')
        return  x,y








