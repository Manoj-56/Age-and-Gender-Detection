from flask import Flask, render_template, request
import numpy as np 
from tensorflow.keras.models import load_model
import cv2

app = Flask(__name__,template_folder='template')

face_cascade = cv2.CascadeClassifier(r'haarcascade_frontalface_default.xml')
model = load_model(r'Detecting_Age.h5')
model2 = load_model(r'gender_Model.h5')
cap = cv2.VideoCapture(0)

@app.route('/')

def man():
	return render_template('detection.html')
    
def AgeDetector(x):
#   read = cv2.imread(x)
  color = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
  size = cv2.resize((cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)),(256,256))
  image = np.array(size)
  res_img = image.reshape(1,256,256,1)
  pre = model.predict(res_img)
  return pre
def Gender(x):
#   read = cv2.imread(x)
  color = cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
  size = cv2.resize((cv2.cvtColor(color,cv2.COLOR_RGB2GRAY)),(300,300))
  image = np.array(size)
  res_img = image.reshape(1,300,300,1)
  pre = model2.predict(res_img)
  return pre
@app.route('/predict',methods = ['POST'])
def home():
    
    while True:
        # Read the frame
        _, img = cap.read()

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (49, 219, 37), 2)
            face_img = img[y:y+h,x:x+w]
            predict_age = AgeDetector(face_img)
            predict_gen = Gender(face_img)
            if np.argmax(predict_age) == 0:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'1-3'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'1-3'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==1:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'4-8'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'4-8'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==2:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'10-15'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'10-15'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==3:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'16-19'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'16-19'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==4:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'20-30'+' '+'Male',(x,y),cv2.FONT_HERSHEY_DUPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'20-30'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==5:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'31-40'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'31-40'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==6:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'40-50'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'40-50'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==7:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'51-60'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                        cv2.putText(img,'51-60'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
            elif np.argmax(predict_age)==8:
                if np.argmax(predict_gen)==0:
                    cv2.putText(img,'61-70'+' '+'Male',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)
                elif np.argmax(predict_gen)==1:
                    cv2.putText(img,'61-70'+' '+'Female',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),2)

        # Display
        cv2.imshow('img', img)

        # Stop if escape key is pressed
        k = cv2.waitKey(30) & 0xff
        if k==27:
            break
            
    # Release the VideoCapture object
    cap.release()
    return 'thanks for checking'


if __name__=="__main__":
	app.run(debug = True)
