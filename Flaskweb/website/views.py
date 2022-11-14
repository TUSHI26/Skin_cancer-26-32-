from flask import Blueprint, render_template,request
from flask_login import login_required, current_user
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow_addons as tfa
from tensorflow import keras
import sys 
from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from . import db
import os

from PIL import Image
#import views.homeimage_location
views = Blueprint('views',__name__)
pred = 7
imag = "2.jpg"            
            
UPLOAD_FOLDER = "C:/xampp/htdocs/Flaskweb/website/static"

@views.route('/', methods=["GET", "POST"])
@login_required
def home():
    if request.method == "POST":
        
           
      image_file = request.files["image"]
     
      global imag 
     
      if image_file:
         image_location = os.path.join(
                UPLOAD_FOLDER,
                image_file.filename
            )
         image_file.save(image_location)
         imag = image_file
         print(image_file)
         print(image_location)
         
            #im = Image.open(image_file)
            #im.show()
         model = tf.keras.models.Sequential()
            # Convolutional & Max Pooling layers
         model.add(tf.keras.layers.Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(128,128,4)))
         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
         model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
         model.add(tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
         model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
            # Flatten & Dense layers
         model.add(tf.keras.layers.Flatten())
         model.add(tf.keras.layers.Dense(512, activation='relu'))
            # performing binary classification
         model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
         model.compile(loss = tfa.losses.SigmoidFocalCrossEntropy(),
         optimizer = tf.keras.optimizers.Adam(),
         metrics = ['binary_accuracy',
         tf.keras.metrics.FalsePositives(),
         tf.keras.metrics.FalseNegatives(),
         tf.keras.metrics.TruePositives(),
            tf.keras.metrics.TrueNegatives()
        ]
)
         test = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img(path = image_location, color_mode = "rgba", target_size = (128,128)), dtype="float32")
         #email = 
         
         
        
         test = test / 255
         test=np.reshape(test,(1,128,128,4))
         test = np.array(test)
         model.load_weights('C:/Users/User/Downloads/weights.hdf5')
         global pred 
         pred= model.predict(test)[0][0]
        
         return render_template("home.html",user=current_user,prediction=pred,imag=imag)
    return render_template("home.html", user=current_user,prediction="",imag=imag)    

@views.route('/action')
def action():

    file = imag.filename
   
    
    
          
    return render_template("action.html",user=current_user,prediction=pred,imag = file )
   
@views.route('/password', methods=["GET", "POST"])
def password():
    
 if request.method == "POST":
     cpass = request.form.get('passID')
     cpass1 = request.form.get('password')
     print(cpass)
     print(cpass1)
     if cpass != cpass1:
            flash('Passwords dont match.', category='error')
     else:
         user1 = current_user.email 
         print(user1)
         first = current_user.first_name
         user = current_user.id
         #conn =SELECT user.id AS user_id, user.email AS user_email, user.password AS user_password, user.first_name AS user_first_name
         #FROM user WHERE user.id = ?]
         user = User.query.get(user)
         db.session.delete(user)
         db.session.commit()
         
         new_user = User(email=user1, first_name=first, password=generate_password_hash(
                 cpass, method='sha256'))
         
         db.session.add(new_user)
         db.session.commit()
         login_user(new_user, remember=True)
         #return redirect(url_for('home'))
 return render_template("password.html",user=current_user)
