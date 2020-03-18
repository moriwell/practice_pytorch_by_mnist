import os
import gzip
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import numpy.random 
import argparse
import sys
import scipy.stats as sci

# Load mhd image
def load_mhd(inputpath, filename, dtype = np.uint8):
   path_tmp = inputpath + '/' + filename + '.mhd'
   img_mhd = sitk.ReadImage(path_tmp)
   nda_img_mhd = sitk.GetArrayFromImage(img_mhd).astype(dtype)                                                            # If size of nda_label is (1024, 256), convert it into (1024, 512)

   return nda_img_mhd

# Save mhd image
def save_mhd(img, outpath, filename, dtype = np.uint8):
   img_mhd = sitk.GetImageFromArray(img)
   img_mhd.SetSpacing([2.8, 2.8])
   path_tmp = outpath + '/' + filename + '.mhd'

   if not os.path.isdir(outpath):
       os.makedirs(outpath)
   sitk.WriteImage(img_mhd, path_tmp)


#main
args = sys.argv
parser = argparse.ArgumentParser(
        description='chainer line drawing colorization')
parser.add_argument('--noise', '-n', default= 'saltpepper',
                        help='add saltpepper or gaussian to GT')
parser.add_argument('--GTsave', '-s',default=True,
                        help='you choose save a GT.True or False')
args = parser.parse_args()

#Data decompression
dl_list = [
    'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz',
]

dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),'raw')
file_path1 = dataset_dir + '/' + dl_list[0]
data1 = gzip.open(file_path1,'rb')
data1= data1.read()
data1 = np.frombuffer(data1,np.uint8,offset=16)

data1.flags.writeable = True
GT_img = data1.reshape(-1,784).copy()
print('Data Loaded')

#add noise
if args.noise == 'saltpepper':
    i = np.random.randint(0, 100, len(data1))
    data1[i < 5] = 255
    data1[i > 95] = 0    

elif args.noise == 'gaussian':
    data1 = sci.zscore(data1)
    c = np.random.randn(47040000,)
    data1 = data1 + c
    ma = max(data1)
    mi = min(data1)
    for i in tqdm(range(len(data1))):
        data1[i]=(data1[i]-mi)/(ma-mi)*255
else:
    print('Error:you have to write -n gaussian or saltpepper as args')
    sys.exit()

print('Noise: {}'.format(args.noise))
noise_img = data1.reshape(-1,784)

#file_path2 = dataset_dir + '/' + dl_list[1]
#data2 = gzip.open(file_path2,'rb')
#data2= data2.read()
#train_label = np.frombuffer(data2,np.uint8,offset=8)

datanum = 10000
outpath = 'J:/git_works/python/step3/AE/data2'


for i in tqdm(range(datanum)):
    if args.GTsave == True:
        save_mhd(GT_img[i].reshape(28,28), outpath + '/GT', 'true{}'.format(i))
    save_mhd(noise_img[i].reshape(28,28), outpath + '/preprocessed3', 'GN{}'.format(i))

