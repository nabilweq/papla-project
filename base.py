from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import scipy
from tensorflow.keras.utils import img_to_array, load_img
from pathlib import Path
import os, os.path

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

source_path = '/Users/nabeeltk/Desktop/Papla project/Datasets/dataset-v2/second/'

path,dirs,files = next(os.walk(source_path))
file_count = len(files)
print(file_count)

original_folder =source_path
    
for y in range(file_count):
  filename = os.listdir(original_folder)[y]
  img_path = original_folder+filename

  img = load_img(img_path)
  x = img_to_array(img)
  x = x.reshape((1,) + x.shape)

  i = 0
  for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='augmented/second', save_prefix='first1', save_format='jpg'):
    i += 1
    if i > 5:
       break
  
  y += 1
  print(y)