import tensorflow.contrib.keras as keras
import os
import numpy as np
data_dir='F:\\line'
datagen=keras.preprocessing.image.ImageDataGenerator(rotation_range=40,
                                                     width_shift_range=0.2,
                                                     height_shift_range=0.2,
                                                     shear_range=0.2,
                                                     zoom_range=0.2,
                                                     horizontal_flip=True,
                                                     vertical_flip=True,
                                                     fill_mode='nearest')
for file in os.listdir(data_dir):
    img=keras.preprocessing.image.load_img(data_dir+'\\'+file)
    x=keras.preprocessing.image.img_to_array(img)
    x=x.reshape((1,)+x.shape)
    i=0
    for batch in datagen.flow(x,batch_size=1,save_to_dir='line',save_prefix='line',save_format='jpeg'):
        i+=1
        if i>4:
            break
