from keras.models import load_model
import numpy as np
from keras.preprocessing import image
import os


classifier = load_model('Trained_model.h5')
classifier.evaluate()

#Prediction of single image
img_name = input('Enter Image Name: ')
image_path =''#image path

# get sign names from already created folders
cwd = os.getcwd()
pd = os.path.dirname(cwd)
test_data_dir = os.path.join(pd,'data\\train')
class_names = sorted(os.listdir(test_data_dir))

#prediction
test_image = image.load_img(image_path, target_size=(200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
predicted_classes = [class_names[np.argmax(pred)] for pred in result]
print(f'Predicted Sign is:   {predicted_classes}')
