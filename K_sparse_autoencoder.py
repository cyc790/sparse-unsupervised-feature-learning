'''
Train the shallow sparse autoencoder on images of dimensions sqrt(m) x sqrt(m)
m = 144 
a = 200 Hidden layer nodes (encoded representation)
m = 144
n = number of images in one batch = 50
Paper title- k-Sparse Autoencoders- Alireza Makhzani , Brendan Frey 
Reference- https://pdfs.semanticscholar.org/b940/43a133e3d07ed0b1cfc036829e619ea0ba22.pdf

'''

import os
from PIL import Image
import csv

def get_im_patches(inputimg, height, width, area):
    im = Image.open(inputimg)
    imgwidth, imgheight = im.size
    totlist = []
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o = o.convert("L")
                row = list(o.getdata())
                totlist.append(row)
            except Exception as e:
                print(e)
            
    #print(totlist)
    im.close()
    return totlist

def save_im_patches(base_dir):
    for f in os.listdir(base_dir):
        if os.path.isfile(base_dir+f):
            if (base_dir+f).lower().endswith('.jpg'):
                patches = get_im_patches(base_dir+f, 12, 12, (0,0,12,12))
                csvfile = open((base_dir+f).replace("jpg", "csv"), 'w')
                wr = csv.writer(csvfile)
                wr.writerows(patches)       #one csv file per image. Contains the 144 values as row for one patch.
                csvfile.close()
        else:
            save_im_patches(base_dir+f+'/')

base_dir="C:/Users/Kunal Phaltane/Downloads/101_ObjectCategories/101_ObjectCategories/"
save_im_patches(base_dir)


from keras.layers import Input, Dense
from keras.models import Model, model_from_json, Sequential, load_model
from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape, Layer, Lambda

import os
from keras.datasets import mnist
import numpy as np
import gc
from PIL import Image

def save_NN_model(filename, model):
    # serialize model to JSON
    model_json = model.to_json()
    with open(filename+'.json', "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(filename+'.h5')
    print("Saved model:"+filename+" to disk")

def load_NN_model(filename):
    # load json and create model
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(filename+'.h5')
    print("Loaded model from disk")
    loaded_model.compile(optimizer='adadelta', loss='mean_squared_error')
    #loaded_model = load_model(filename)
    return loaded_model

def Ksparse(X):
    #choose top k = 100 points from the representation.
    #print(type(X))    numpy array
    k = 200
    temp = np.copy(X)
    indices = temp.argsort()[-k:][::-1]
    for i in range(temp.size):
        if i not in indices:
            X[i] = 0.0
    print(X)
    return X

def CreateEncoder():
    enc = Sequential()
    enc.add(Dense(200, input_shape=(144,) ) )       #200 nodes in hidden layer representation
    enc.add(Activation('sigmoid'))
    enc.add(Lambda(function=Ksparse))
    return enc

def CreateDecoder():
    dec = Sequential()
    dec.add(Dense(144, input_shape=(200, ) ))     #output layer = 144x x.. input will be calculated at runtime.
    dec.add(Activation('sigmoid'))
    return dec


enc, dec = CreateEncoder(), CreateDecoder()
AE = Sequential()
AE.add(enc)
AE.add(dec)
AE.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy','mean_absolute_error','mean_squared_error'])



base_dir = "C:/Users/Kunal Phaltane/Downloads/"
patch_counter=0

Allfiles = os.listdir(base_dir+"airplanes")
files =[]
for f in Allfiles:
    if f.lower().endswith('.csv'):
        files.append(base_dir+"airplanes/"+f)

if len(files) > 0:
    batch = np.genfromtxt(files[len(files)-1], delimiter=',')
    batch = batch.astype('float32')/255.
    batch.reshape((len(batch), np.prod(batch.shape[1:])))
    files = files[:len(files)-1]  #remove last element.
else:
    print("Error")

EndOfFiles=0


for i in range(100):
    while batch.shape[0] <= 20000:
        if len(files) > 0:
            print("Adding to batch")
            foo = np.genfromtxt(files[len(files)-1], delimiter=',')
            foo = foo.astype('float32')/255.
            foo.reshape((len(foo), np.prod(foo.shape[1:])))
            files = files[:len(files)-1]  #remove last element.
            batch = np.concatenate((batch, foo))
        else:
            EndOfFiles=1    #set flag
            break
    #end of while
    if EndOfFiles == 1:
        break       #files are over so get out of for loop
        print("Files are over")
    else:
        #train the encoder
        print("Training model with batch size", batch.shape[0])
        AE.fit(batch, batch, batch_size=256, nb_epoch=50, verbose=1)
        if len(files) > 0:
            print("Creating new batch")
            batch = np.genfromtxt(files[len(files)-1], delimiter=',')   #start new batch
            batch = batch.astype('float32')/255.
            batch.reshape((len(batch), np.prod(batch.shape[1:])))
            files = files[:len(files)-1]  #remove last element.

try:
    AE.save("C:/Users/Kunal Phaltane/Downloads/my_model.h5") #not saving
except Exception as e:
    print(e)

x_test = np.genfromtxt("C:/Users/Kunal Phaltane/Downloads/image_0799.csv", delimiter=',')
reconstructed = AE.predict(x_test)
reconstructed=reconstructed*255
recontructed = reconstructed.astype('uint8')


x_im = Image.open("C:/Users/Kunal Phaltane/Downloads/image_0799.jpg")
imw,imh = x_im.size
x_im.close()

tile_no_rows, tile_no_cols = int(imh/12), int(imw/12)

print (reconstructed.shape)

im_n = Image.new('L', (tile_no_cols*12, tile_no_rows*12) )

counter = 0
flag = 0

for i in range(0, tile_no_rows):
	for j in range(0, tile_no_cols):
		if counter <=x_test.shape[0]:
			im_n.paste( Image.fromarray(reconstructed[counter].reshape(12,12)), (j*12, i*12))
			counter = counter + 1


im_n.save("C:/Users/Kunal Phaltane/Downloads/image_0799_rec.jpg")

#loss: 0.0012 - acc: 0.1871 - mean_absolute_error: 0.0227 - mean_squared_error: 0.0012


