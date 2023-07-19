import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import io


# LINK TO THE CSS FILE
with open(r"C:\Users\Rishitha Reddy\OneDrive\Desktop\Brain_Tumor\style.css")as f:
 st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html = True)


def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://cdn.pixabay.com/photo/2019/04/24/11/27/flowers-4151900_960_720.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url() 



# @st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model(r'C:\Users\Rishitha Reddy\OneDrive\Desktop\Brain_Tumor\models\model_cnn.h5')
  return model
with st.spinner('Model is being loaded..'):
  model=load_model()

st.write("""
         # Brain Tumor Classification
         """
         )

file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])

st.set_option('deprecation.showfileUploaderEncoding', False)

def scale(image):
    image = tf.cast(image ,tf.float32)
    image /= 255.0

    return image

# def import_and_predict(image_data, model):
    
#         opencvImage = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
#         img = cv2.resize(opencvImage,(256,256))
#         img = img.reshape(1,256,256,3)
        
#         img_reshape = scale(image)
    
#         prediction = model.predict(img_reshape)
        
#         return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image1 = Image.open(file)
    image = cv2.cvtColor(np.array(image1), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image,(256,256))
    # image = scale(image)
    image = tf.reshape(image, [1, 256, 256,3])
    prediction = model.predict(image)
    st.image(image1, use_column_width=True)
    # st.write(prediction)
    score = tf.nn.softmax(prediction[0])
    st.write("prediction")
    class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    # st.write(score)
    
    st.write(class_names[np.argmax(score)])
    # st.write(prediction[np.argmax(score)])
    # 100 * np.max(score)
