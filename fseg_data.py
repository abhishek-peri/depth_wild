import os
from absl import app
from absl import flags
from absl import logging
import numpy as np
import cv2
import os, glob

#import alignment
#from alignment import compute_overlap
#from alignment import align


SEQ_LENGTH = 3
WIDTH = 416
HEIGHT = 128
STEPSIZE = 1
#INPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_FULL/kitti-raw-uncompressed'
#OUTPUT_DIR = '/usr/local/google/home/anelia/struct2depth/KITTI_procesed/'


def get_line(file, start):
    file = open(file, 'r')
    lines = file.readlines()
    lines = [line.rstrip() for line in lines]
    ret = None
    for line in lines:
        nline = line.split(': ')
        if nline[0]==start:
            ret = nline[1].split(' ')
            ret = np.array([float(r) for r in ret], dtype=float)
            ret = ret.reshape((3,4))[0:3, 0:3]
            break
    file.close()
    return ret


def run_all(INPUT_DIR,OUTPUT_DIR):
    ct = 0
    # 
    if not OUTPUT_DIR.endswith('/'):
        OUTPUT_DIR = OUTPUT_DIR + '/'

    #list d contains folders of all dates
    for d in glob.glob(INPUT_DIR + '/*/'):
        print(d)
        date = d.split('/')[-2] # one of the date folders
        #print(date)
        #file_calibration = d + 'calib.txt' # the calibration file
        #print(file_calibration)
        #calib_raw = [get_line(file_calibration, 'P2'), get_line(file_calibration, 'P3')] # get the P2 and P3 instrinsic matrices
        #print(calib_raw)

        
        #print(succ)

        #d2 is list of subfolders containign sequences 0 to 20 in a given dates folder
        #for d2 in glob.glob(d + '*/'):
            #print(d2)
        seqname = d.split('/')[-2]
        print(seqname)
        print('Processing sequence', seqname)
        ct = 1
        if not os.path.exists(OUTPUT_DIR + seqname):
            os.mkdir(OUTPUT_DIR + seqname)
        folder = d
        files = glob.glob(folder + '/*.png')
        #print(files)
        #files = [file for file in files if not 'disp' in file and not 'flip' in file and not 'seg' in file]
        files = sorted(files)
        #print(files)
        #print(succ)
        for i in range(SEQ_LENGTH, len(files)+1, STEPSIZE):
            #print('success')
            imgnum = str(ct).zfill(10)
            #print(imgnum)
            if os.path.exists(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png'):
                ct+=1
                continue
            big_img = np.zeros(shape=(HEIGHT, WIDTH*SEQ_LENGTH, 3))
            wct = 0

            for j in range(i-SEQ_LENGTH, i):  # Collect frames for this sample.
                img = cv2.imread(files[j])
                ORIGINAL_HEIGHT, ORIGINAL_WIDTH, _ = img.shape
                #print(WIDTH)
                #print(ORIGINAL_WIDTH)

                zoom_x = float(WIDTH)/ORIGINAL_WIDTH
                zoom_y = float(HEIGHT)/ORIGINAL_HEIGHT
                
                

                img = cv2.resize(img, (WIDTH, HEIGHT))
                big_img[:,wct*WIDTH:(wct+1)*WIDTH] = img
                wct+=1
            cv2.imwrite(OUTPUT_DIR + seqname + '/' + imgnum + '-fseg.png', big_img)
            #f = open(OUTPUT_DIR + seqname + '/' + imgnum + '_cam.txt', 'w')
            #f.write(calib_representation)
            #f.close()
            ct+=1

def main(_):
  INPUT_DIR = '/media/abhishek/Elements/datasets/kitti_fseg/sequences'
  OUTPUT_DIR = '/media/abhishek/Elements/datasets/KITTI_fseg_procesed/'
  run_all(INPUT_DIR,OUTPUT_DIR)


if __name__ == '__main__':
  app.run(main)
