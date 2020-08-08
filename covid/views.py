from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image

# Create your views here.
model = keras.models.load_model('./model/new_model.hdf5')
def index(request):
    return render(request,'index.htm')



def predict(request):
    #print(request.POST.dict())
    #print(request.FILES['xray'].name)
    myfile = request.FILES['xray']
    fs = FileSystemStorage()
    filename = fs.save(myfile.name, myfile)
    uploaded_file_url = fs.url(filename)
    fileurl='.'+uploaded_file_url
    test_img=image.load_img(fileurl, target_size=(224,224))
    img = image.img_to_array(test_img)
    img = img/225
    img = img.reshape(1,224,224,3)
    #print(img)
    result=model.predict(img)
    #print(result)
    res = ''
    color = ''
    if result[0][0] > result[0][1]:
        res = "Positive"
        color ='red'
    else:
        res = "Negetive"
        color ='green'
    return render(request,'predict.htm', {
            'url': uploaded_file_url, "result":res, "color":color
        })
