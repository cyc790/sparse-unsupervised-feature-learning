from PIL import Image
import csv

def getcsvdata( inputimg, height, width, k, area):
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
                o.save(os.path.join("%s" % inputimg,"Part-%s.png" % k))
            except:
                pass
            k +=1
    print(totlist)
    csvfile = open(inputimg.replace("jpg","csv"), 'w')
    wr = csv.writer(csvfile)
    wr.writerows(totlist)
    csvfile.close()

getcsvdata("cockroach.jpg",12,12,1,(0,0,12,12))
