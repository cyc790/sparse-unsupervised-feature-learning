'''
Supply a base directory and it gives out csv corresponding to images in that directory structure
with each row representing one patch flattened into a row of 144 pixels.
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
