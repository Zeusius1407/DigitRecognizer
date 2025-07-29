import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("Handwritten Digit Recognizer")

st.markdown("Draw a digit (0–9) below and click **Recognize**.")

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
    if canvas_result.image_data is None:
        st.warning("⚠️ Canvas is empty. Please draw a digit.")
    else:
        img_data = canvas_result.image_data

        try:
            # Convert to grayscale (average RGB if needed)
            if img_data.shape[2] == 3:  # RGB
                img_gray = np.mean(img_data, axis=2)
            else:  # Already grayscale
                img_gray = img_data[:, :, 0]

            # Check if the image is blank
            if np.mean(img_gray) > 450:
                st.warning("⚠️ The canvas looks blank. Please draw a digit.")
            else:
                # Convert to PIL image
                img = Image.fromarray(img_gray.astype(np.uint8))
                img = img.resize((28, 28)).convert("L")
                img_array = np.array(img)
                img_array = 255 - img_array  # Invert: white background, black digit
                img_array = img_array / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # Make prediction
                prediction = model.predict(img_array)
                digit = np.argmax(prediction)

                st.success(f"Recognized Digit: **{digit}**")

        except Exception as e:
            st.error(f"⚠️ Failed to process the image.\n\nError: {e}")
