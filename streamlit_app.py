import streamlit as st
import pickle
import tensorflow as tf
import numpy as np
from PIL import Image
model = tf.keras.models.load_model('my_model.h5',  compile=False)

# def predict(model, img):
#     img = Image.open(img).resize([256,256])
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img, axis=0) 

#     prediction = model.predict(img_array)

#     class_names = ['Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust']
#     predicted_class = class_names[np.argmax(prediction[0])]

#     return predicted_class

def predict(model, img):
    preserve_alpha_channel = False
    img = Image.open(img).resize([256, 256])
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # Handle different image formats and alpha channels
    if img_array.shape[-1] == 4:  # Alpha channel present
        if preserve_alpha_channel:  # If model requires alpha channel
            img_array = img_array.astype(np.float32) / 255.0  # Normalize
        else:
            img_array = img_array[:, :, :3]  # Extract RGB channels

    # Normalize pixel values if not already done
    img_array = img_array.astype(np.float32) / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_names = ['Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust']
    predicted_class = class_names[np.argmax(prediction[0])]

    return predicted_class

html_temp = """ <div style="background-color:#290F0F;padding:10px">
    <h2 style="color:white;text-align:center;">Coffee Crop Disease Detection</h2>

    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

html_temp2 = """ <div style="background-color:#290F0F;padding:10px">
    <h6 style="color:white;text-align:center;">This application detects if the coffee leaf is healthy, rusted or contains spider mite</h6>
    <h6 style="color:white;text-align:center;">Please upload clear coffee leaf image for accurate results</h6>
    </div>
    """
st.write(html_temp2,unsafe_allow_html=True)

choose_img = st.selectbox('Pick one', ['Upload Image', 'Take a picture'])

if choose_img == 'Upload Image':
    input_img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if input_img is not None:
        st.image(input_img, width=700)
else:
    input_img = st.camera_input("Take a picture") 



result=""
if st.button("Predict"):
    result=predict(model, input_img)
    st.success(result)



