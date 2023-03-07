#!/usr/bin/env python
# coding: utf-8

# In[3]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[4]:


from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob


# In[5]:


IMAGE_SIZE = [224,224]

train_path = 'dataset/train'
valid_path = 'dataset/test'


# In[6]:


resnetn= ResNet50(input_shape=IMAGE_SIZE + [3],weights = 'imagenet', include_top=False)


# In[8]:


for layer in resnetn.layers:
    layer.trainable = False


# In[9]:


folders = glob('dataset/train/*')


# In[11]:


x = Flatten()(resnetn.output)


# In[12]:


prediction =Dense(len(folders), activation = 'softmax')(x)

model= Model(inputs=resnetn.input, outputs=prediction)


# In[13]:


model.summary()


# In[15]:


model.compile(
    loss = 'categorical_crossentropy',
    optimizer = 'adam',
    metrics=['accuracy']
)


# In[26]:


train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range =0.2,
    zoom_range = 0.2, 
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1./255)


# In[27]:


training_set = train_datagen.flow_from_directory('dataset/train',
                                              target_size = (224,224),
                                              batch_size = 32,
                                              class_mode = 'categorical')


# In[28]:


test_set = test_datagen.flow_from_directory('dataset/test',
                                              target_size = (224,224),
                                              batch_size = 32,
                                              class_mode = 'categorical')


# In[21]:


len(test_set)


# In[29]:


r = model.fit_generator(
    training_set,
    validation_data=test_set,
    epochs=20,
    steps_per_epoch=len(training_set),
    validation_steps = len(test_set)
)


# In[ ]:




