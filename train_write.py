import glob
import os

f = open('./train.txt', 'w')

folders = glob.glob('./'+'/*/')

for item in folders:
    for i in range(1,len(os.listdir(item))/3 +1):
        temp1 = item.split('/')[-2]
        temp2 = str(i).zfill(10)
        f.write(temp1 + ' ' + temp2 + '\n')
        #f.write('/n')
f.close()