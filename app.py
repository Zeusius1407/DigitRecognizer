import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("Handwritten Digit Recognizer")

st.markdown("Draw a digit (0–9) below and click **Predict**.")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=7,
    stroke_color="black",
    background_color="white",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Recognize"):
    if canvas_result.image_data is not None and np.mean(canvas_result.image_data[:, :, 0]) < 350:
        img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
        img = img.resize((28, 28)).convert('L')
        img_array = np.array(img)
        img_array = 255 - img_array  
        img_array = img_array / 255.0 
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        digit = np.argmax(prediction)

        st.write(f"Recognized Digit: **{digit}**")
    else:
        st.warning("Please draw a digit first.")
