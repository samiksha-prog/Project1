import streamlit as st
import numpy as np
import pickle
import cv2
import os
from PIL import Image
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input

st.title('Acrylamide Detection in Potato Chips')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

file_name = file_selector("Dataset/test")
st.write('You selected `%s`' % os.path.abspath(file_name))

# loading svm model
model = pickle.load(open('model.pkl','rb'))

# loading vgg16 model
# load json and create model
json_file = open('vgg_model.json', 'r')
vgg_model_json = json_file.read()
json_file.close()
vgg_model = model_from_json(vgg_model_json)
# load weights into new model
vgg_model.load_weights("vgg_model.h5")

bt = st.button('Click to predict')
if bt:
    image = Image.open(file_name)
    st.columns(3)[1].image(image, caption='Potato Chip', width=250)

    img = cv2.imread(file_name)  #input image over here
    # ROI Segmentation
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Set minimum area threshold
    min_area = 50

    # Iterate through contours
    for contour in contours:
        # Calculate area
        area = cv2.contourArea(contour)

        # Remove contour if area is below threshold
        if area < min_area:
            cv2.drawContours(thresh, [contour], 0, 0, -1)

    # Apply mask to original image
    result = cv2.bitwise_and(img, img, mask=thresh)
    cv2.imwrite('./Dataset/test.jpg', result)

    # constructing feature matrix using VGG16 model
    def get_features(img_path):
        img = load_img(img_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        flatten = vgg_model.predict(x)
        return list(flatten[0])

    features_t = []
    features_t.append(get_features('Dataset/test.jpg'))

    # predicting the output using SVM
    predicted = model.predict(features_t)[0]

    st.warning("Output: ")
    if predicted==1:
        st.write("Acrylamide is present in the chips.")
    else:
        st.write("Acrylamide is not present in the chips.")
    st.success('Done!')