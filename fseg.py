import cv2
import os
import glob
from PIL import Image

path_dir = '/media/abhishek/Elements/datasets/kitti_dataset_new/sequences'
out_dir = '/media/abhishek/Elements/datasets/kitti_fseg/sequences'
folders = ['06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21']

for item in folders:
    img_path = os.path.join(path_dir,item,'image_3_fseg')
    img_list = glob.glob(img_path+'/*.png')
    frame_list = os.listdir(img_path)
    print('processing sequence {}'.format(item))
    for i in range(len(img_list)):
        #img = cv2.imread('/media/abhishek/Elements/datasets/kitti_dataset_new/sequences/06/image_3_fseg/000063.png',0)
        img = cv2.imread(img_list[i],0)
        ret,thresh1 = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(out_dir,item,frame_list[i]),thresh1)
        #print(succ)
        #/media/abhishek/Elements/datasets/kitti_dataset_new/sequences/06