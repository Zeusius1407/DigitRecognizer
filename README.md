# Handwritten Digit Recognizer
A machine learning model which will recognize the digit the user has drawn on a
canvas.

# Tech Stack Used

User Interface: streamlit\
Canvas: streamlit-drawable-canvas\
Image handling: pillow and numpy\
ML Model: tensorflow\
Scripting: Python

# Working
The model is a Convolusional Neural Network (CNN) consisting of 4 blocks of
Convolusion layers followed by a fully connected layer.\
The dataset used to train the CNN is [MNIST](https://www.tensorflow.org/datasets/catalog/mnist) which consists of large amounts of
hand written digits.\
The script of the neural net is present in model.py.\
The user is presented with a canvas where they can draw any digit they want,
the image is then processed and used to make prediction by the model, this
prediction is then presented to the user.

# Instructions

Download all the files present in this repository.\
Make sure all the required modules are installed.\
Execute the model.py script by running this command in the terminal:
```
python model.py
```
After the script has been executed run the following command in the terminal:
```
streamlit run app.py
```
This will open a new tab in the user's browser containing a blank white canvas
for the user to draw on.\
Clicking on the Recognize button will present the user with the model's
prediction of the digit that the user has drawn. 
