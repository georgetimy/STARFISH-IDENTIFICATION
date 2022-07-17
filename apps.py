from cgitb import html
from re import X
from PIL import Image
from matplotlib import pyplot as plt
import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="STARFISH IDENTIFICATION")

st.title('STARFISH IDENTIFICATION')

instructions = """
        Either upload your own image or select from
        the sidebar to get a preconfigured image.
        The image you select or upload will be fed
        through the Convolutional Neural Network in real-time
        and the output will be displayed to the screen.
        """
st.write(instructions)

st.subheader('Here is the example image')

st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
     width: 280px
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
    width: 280px
    margin-left; -280px
}
</style>

""",
unsafe_allow_html=True,
)

st.sidebar.title('STARFISH SIDEBAR')
st.sidebar.subheader('Here the starfish that you can identify')

def main():
    selected_box = st.sidebar.selectbox("Select Starfish:",
                    ("Culcita Novaeguineae", "Linckia Laevigata", "Luidia Foliolata", "Protoreaster Nodosus"))

    if selected_box == 'Culcita Novaeguineae': st.image(image = Image.open('Cushionstar.jpg')) 

    if selected_box == 'Linckia Laevigata': st.image(image = Image.open('Bluestar.jpg'))

    if selected_box == 'Luidia Foliolata': st.image(image = Image.open('Sandstar.jpg'))

    if selected_box == 'Protoreaster Nodosus': st.image(image = Image.open('Hornedseastar.jpg'))

if __name__ == "__main__":
        main()

def main():
        html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">UPLOAD STARFISH</h2>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

if __name__ == "__main__":
        main()

st.subheader("The image will be automatically identified")

def main():
        uploaded_file = st.file_uploader("Choose a file", type= ['jpg', 'png', 'jpeg'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            figure = plt.figure()
            plt.imshow(image)
            plt.axis('off')
            result = predict_class(image)
            st.subheader(result)
            st.pyplot(figure)

def preprocess(img, input_size):
        nimg = img.convert('RGB').resize(input_size, resample= 0)
        img_arr = (np.array(nimg))/255
        return img_arr

def reshape(imgs_arr):
        return np.stack(imgs_arr, axis=0)

def predict_class(image):
        cnn_model = tf.keras.models.load_model('model.h5')
        input_size = (32, 32)
        class_names = ['Culcita Novaeguineae ',
                        'Linckia Laevigata',
                        'Luidia Foliolata',
                        'Protoreaster Nodosus']
        X = preprocess(image,input_size)
        X = reshape([X])
        predictions = cnn_model.predict(X)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        result = "The image is identified as: {}".format(image_class)

        return result

if __name__ == "__main__":
        main()

