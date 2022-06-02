import pdb
import sys
import os
import cv2
import numpy as np
import torch
import threading
from tqdm import tqdm
sys.path.append('./avalanche')
from devkit_tools import ChallengeDetectionDataset, ChallengeClassificationDataset
from examples.tvdetection.transforms import RandomZoomOut
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt


def show_im(cls_dataset, n, cnt):
    for i in range(n):
        im, lbl = cls_dataset[i]
        # h,w = im.height, im.width
        # if w < 224 and h < 224:
            # im = trance(im)[0]
        #pdb.set_trace()
        im = cv2.cvtColor(np.asarray(im),cv2.COLOR_RGB2BGR)
        print(f'size: {im.shape}')
        
        if im.shape[0] <= 224 and im.shape[1] <= 224:
            cnt += 1
        im = cv2.resize(im, (224,224))
        
        cv2.putText(im,"label" + str(lbl),(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA)
        cv2.imshow('train', im)
        cv2.waitKey(50)
    print(f'radio:{cnt}/{n}')

def mean_std(cls_dataset, n):
    means = torch.zeros(3)
    stds = torch.zeros(3)
    for i in tqdm(range(n)):
        im, lbl = cls_dataset[i]
        im = F.pil_to_tensor(im)
        im = F.convert_image_dtype(im)
        for d in range(3):
            means[d] += im[d, :, :].mean()
            stds[d] += im[d, :, :].std()
    means.div_(len(cls_dataset))
    stds.div_(len(cls_dataset))
    print(f'mean:{list(means.numpy())}\tstd:{list(stds.numpy())}')

def bbox_mean_std(cls_dataset, n):
    ratio_list = []
    for i in tqdm(range(n)):
        _, lbl = cls_dataset[i]
        bbox = lbl['boxes'].numpy()
        for bb in bbox:
            ratio_list.append((bb[2] - bb[0])/(bb[3] - bb[1]))
    
    ratio_array = np.array(ratio_list)
    ratio_max = np.max(ratio_array)
    ratio_min = np.min(ratio_array)
    ratio_mean = np.mean(ratio_array)
    ratio_var = np.var(ratio_array)
    print(f'max:{ratio_max}\t min:{ratio_min}\t mean:{ratio_mean}\t var:{ratio_var}')
    plt.figure()
    plt.hist(ratio_array,20)
    plt.xlabel('Ratio of length / width')
    plt.ylabel('Frequency of ratio')
    plt.title('Ratio\n' \
          +'max='+str(round(ratio_max,2))+', min='+str(round(ratio_min,2))+'\n' \
          +'mean='+str(round(ratio_mean,2))+', var='+str(round(ratio_var,2))
          )
    plt.show()



def index_(cls_dataset, n):
    index_list = []
    for i in tqdm(range(n)):
        im, _ = cls_dataset[i]
        w,h = F.get_image_size(im)
        if True:
            index_list.append(i)
    print(f'mean:{index_list}\tstd:{len(index_list)}')


if __name__ == "__main__":
    root = '/media/datd/fht0/cvprw/datasets/'
    im_path = root + '/images'
    train_json = root + 'split_ego_train.json'#'/ego_objects_challenge_train.json'
    show_images = False
    try_loading = False
    train = True
    instance_level = True




    cls_dataset = ChallengeClassificationDataset(root=root,
                train=train,
                bbox_margin=20,
                instance_level=instance_level)


    cls_dataset = ChallengeDetectionDataset(root=root,
                train=train,
                )

    n = len(cls_dataset)
    cnt = 0
    print(f"train set:{n}")
    index_(cls_dataset, n)