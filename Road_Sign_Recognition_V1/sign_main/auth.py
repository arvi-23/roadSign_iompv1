#Importing Flask Modules
from flask import Flask
app = Flask(__name__)
from flask import render_template ,request 
from werkzeug.utils import secure_filename

# Importing Tensorflow Modules
import wikipedia
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import numpy as np
import os 
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
  
model = tf.keras.models.load_model('E:\Road_Sign_Recognition_V1\sign_main\m1.h5')

import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
import tensorflow
import numpy as np 
    

def prepare(img_path):
    img = tensorflow.keras.utils.load_img(img_path, target_size=(30,30))
    x = tensorflow.keras.utils.img_to_array(img)
    return np.expand_dims(x, axis=0)

def finds(image_path):
    classes = ['Speed limit (20km/h)',
            'Speed limit (30km/h)', 
            'Speed limit (50km/h)', 
            'Speed limit (60km/h)', 
            'Speed limit (70km/h)', 
            'Speed limit (80km/h)', 
            'End of speed limit (80km/h)', 
            'Speed limit (100km/h)', 
            'Speed limit (120km/h)', 
            'No passing', 
            'No passing veh over 3.5 tons', 
            'Right-of-way at intersection', 
            'Priority road', 
            'Yield', 
            'Stop', 
            'No vehicles', 
            'Veh > 3.5 tons prohibited', 
            'No entry', 
            'General caution', 
            'Dangerous curve left', 
            'Dangerous curve right', 
            'Double curve', 
            'Bumpy road', 
            'Slippery road', 
            'Road narrows on the right', 
            'Road work', 
            'Traffic signals', 
            'Pedestrians', 
            'Children crossing', 
            'Bicycles crossing', 
            'Beware of ice/snow',
            'Wild animals crossing', 
            'End speed + passing limits', 
            'Turn right ahead', 
            'Turn left ahead', 
            'Ahead only', 
            'Go straight or right', 
            'Go straight or left', 
            'Keep right', 
            'Keep left', 
            'Roundabout mandatory', 
            'End of no passing', 
            'End no passing veh > 3.5 tons' ]

    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    result = model.predict([prepare(image_path)])
    
    disease=tensorflow.keras.utils.load_img(image_path)

    
    classresult=np.argmax(result,axis=1)
    return classes[classresult[0]] 

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("base.html") 

@app.route('/mainpage', methods=['GET', 'POST'])
def home123():
    return render_template("mainpage.html") 


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        print(f.filename)
        result=finds(file_path)  
        try:
            if wikipedia.page(result) is None:
                print("no")
                
            else:
                ans=wikipedia.summary(result,sentences=1)
                nothing= wikipedia.page(result).url
            # return render_template("Detect.html",task1=f.filename,tasks=result,ans=ans,nothing=nothing)
                return render_template("detect.html",task1=f.filename,tasks=result,ans=ans,nothing=nothing)
        except:
            ans="not found"
            return render_template("detect.html",task1=f.filename,tasks=result,ans=ans)

    else:
        return  render_template("detect.html")      



import pyttsx3


def text_to_speech(text, gender):
    """
    Function to convert text to speech
    :param text: text
    :param gender: gender
    :return: None
    """
    voice_dict = {'Male': 0, 'Female': 1}
    code = voice_dict[gender]

    engine = pyttsx3.init()

    # Setting up voice rate
    engine.setProperty('rate', 125)

    # Setting up volume level  between 0 and 1
    engine.setProperty('volume', 0.8)

    # Change voices: 0 for male and 1 for female
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[code].id)

    engine.say(text)
    engine.runAndWait()


@app.route('/speech', methods=['POST', 'GET'])
def homepage1():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        basepath, 'static', secure_filename(f.filename))
        f.save(file_path)
        print(file_path)
        print(f.filename)
        result=finds(file_path)
        gender = request.form['voices']
        text_to_speech(result, gender)
        return render_template('speech.html',task1=f.filename)
    else:
        return render_template('speech.html')















if __name__ == '__main__':
    app.run(debug=True)
