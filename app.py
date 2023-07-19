from flask import Flask, request, render_template, url_for, Response
from flask.helpers import send_file
import cv2
import os
import keras
from tensorflow.keras.models import model_from_json
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)



@app.route('/')
def home():
	return render_template("index.html")
	
@app.route('/upload', methods=['GET','POST'])	
def upload():
	global photo_path
	photo_path = None
	if request.method == 'POST':
		uploaded_file = request.files['image']
				
		
		if not uploaded_file:
			return render_template("index.html")

		
		uploaded_file.save('static/uploads/'+uploaded_file.filename)
		
		photo_path = 'static/uploads/'+uploaded_file.filename
		
		return render_template('index.html', photo_path = photo_path)
		
@app.route('/capture', methods=['GET','POST'])	
def capture():
	global image_path
	image_path = None
	
	cap =cv2.VideoCapture(0)
	
	if not cap.isOpened():
		return render_template("index.html")
	
	_,frame = cap.read()
	
	cap.release()

	image_path='static/uploads/'
	os.makedirs(image_path, exist_ok = True)
	image_path = os.path.join(image_path, 'captured_photo.jpeg')
	cv2.imwrite(image_path,frame)
		
	return render_template('index.html', image_path = image_path)	
		

@app.route('/predict', methods=['GET', 'POST'])
def predict():
	
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	loaded_model = model_from_json(loaded_model_json)
	
	loaded_model.load_weights("model.h5")
	
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	if photo_path == None:
		return render_template('index.html', photo_path = photo_path)
		
	img = cv2.imread(photo_path)	
	#gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	
	faces = facec.detectMultiScale(img, 1.1, 5)
	if faces == ():
		return render_template('error.html', photo_path = photo_path)
	prediction=None
	
	for x,y,w,h in faces:
		face_roi = img[y:y+h, x:x+w]
		
		global res
		res=0
		resized_img = cv2.resize(face_roi, (64, 64))
		print(resized_img.shape)
		resized_img = resized_img / 255.0
		plt.imshow(resized_img)
		plt.axis('off')  # Optional: Turn off axis labels
		plt.show()
			
			
		prediction = loaded_model.predict(np.expand_dims(resized_img,axis=0))
		res = prediction.item()
		res = round(res,2)
		print("----------------------------------------")
		print(res)
		print("----------------------------------------")
			
			
	if res >= 0.59:
		return render_template('nonwrinkle.html', photo_path = photo_path)
		
	else:
		return render_template('wrinkle.html', photo_path = photo_path)
	

@app.route('/predict_cap', methods=['GET', 'POST'])
def predict_cap():
	
	
	json_file = open('model.json','r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	loaded_model = model_from_json(loaded_model_json)
	
	loaded_model.load_weights("model.h5")
	
	loaded_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
	facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

	if image_path == None:
		return render_template('index.html', image_path = image_path)
		
	img = cv2.imread(image_path)	
	gray_image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	faces = facec.detectMultiScale(gray_image, 1.3, 5)
	if faces == ():
		return render_template('error.html', image_path = image_path)
	prediction=None
	for x,y,w,h in faces:
		face_roi = img[y:y+h, x:x+w]
		
		global res
		res=0
		resized_img = cv2.resize(face_roi, (64, 64))
		print(resized_img.shape)
		resized_img = resized_img / 255.0
		
		plt.imshow(resized_img)
		plt.axis('off')  # Optional: Turn off axis labels
		plt.show()
			
			
		prediction = loaded_model.predict(np.expand_dims(resized_img,axis=0))
		res = prediction.item()
		res = round(res,2)
		print("----------------------------------------")
		print(res)
		print("----------------------------------------")
			
			
			
			
			
	if res >= 0.6:
		return render_template('nonwrinkle.html', image_path = image_path)
		
	else:
		return render_template('wrinkle.html', image_path = image_path)
	

	
if __name__ == "__main__":
	app.run(debug=True)
