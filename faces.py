import cv2
import os
import numpy as np
import pickle
import glob
import face_recognition
import time
import imutils
import sys
import shutil
from PIL import Image

def is_similar(last_item):
	unknown_image = face_recognition.load_image_file('images/unknown/'+str(last_item)+'.jpg')
	unknown_face_encoding = face_recognition.face_encodings(unknown_image)
	if len(unknown_face_encoding) > 0:
		print(last_item)
		image_1 = face_recognition.load_image_file('images/unknown/'+str(last_item-1)+'.jpg')
		image_compare = face_recognition.face_encodings(image_1)
		print('images/unknown/'+str(last_item)+'.jpg')
		print('images/unknown/'+str(last_item-1)+'.jpg')
		if len(image_compare) > 0:
			image_1_encoding = image_compare[0]
			unknown_face_encoding=unknown_face_encoding[0]
			results = face_recognition.compare_faces(
				[image_1_encoding], unknown_face_encoding)
			time.sleep(3)
			if results[0]:
				return True
			else:
				return False
		else :
			os.remove('images/unknown/'+str(last_item)+'.jpg')
	else :
		os.remove('images/unknown/'+str(last_item)+'.jpg')
	time.sleep(3)
	return False

def recognition():
	face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("./recognizers/face-trainner.yml")
	with open("pickles/face-labels.pickle", 'rb') as f:
		og_labels = pickle.load(f)
		labels = {v:k for k,v in og_labels.items()}
	cap = cv2.VideoCapture(0)
	while(True):
		ret, frame = cap.read()
		gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
		for (x, y, w, h) in faces:
			roi_gray = gray[y:y+h, x:x+w] 
			roi_color = frame[y:y+h, x:x+w]
			id_, conf = recognizer.predict(roi_gray)
			if conf>=4 and conf <= 85:
				print(conf)
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = labels[id_]
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
			else :
				print(conf)
				font = cv2.FONT_HERSHEY_SIMPLEX
				name = "unknown"
				color = (255, 255, 255)
				stroke = 2
				cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
				last_item = int(sorted(glob.glob("images/unknown/*.jpg"),key=os.path.getmtime)[-1].split("\\")[1].split(".")[0])+1
				img_item = "images/unknown/"+str(last_item)+".jpg"
				cv2.imwrite(img_item, roi_gray)
				if is_similar(last_item):
					os.remove(img_item)
					print('true')
			color = (255, 0, 0) 
			stroke = 2
			end_cord_x = x + w
			end_cord_y = y + h
			cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
		cv2.imshow('frame',frame)
		if cv2.waitKey(20) & 0xFF == ord('q'):
			break		
	cap.release()
	cv2.destroyAllWindows()

def train():
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	image_dir = os.path.join(BASE_DIR, "videos")
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("mp4") or file.endswith("avi"):
				path = os.path.join(root, file)
				file_name = file.split('.')
				directory = "images/"+file_name[0]
				if os.path.exists(directory):
					shutil.rmtree(directory)
				os.makedirs(directory)
				cap= cv2.VideoCapture(path)
				i=0
				while(cap.isOpened()):
					ret, frame = cap.read()
					if ret == False:
						break
					cv2.imwrite(directory+"/"+str(i)+'.jpg',frame)
					i+=1
				cap.release()
				cv2.destroyAllWindows()

	image_dir = os.path.join(BASE_DIR, "images")
	face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	current_id = 0
	label_ids = {}
	y_labels = []
	x_train = []
	for root, dirs, files in os.walk(image_dir):
		for file in files:
			if file.endswith("png") or file.endswith("jpg"):
				path = os.path.join(root, file)
				label = os.path.basename(root).replace(" ", "-").lower()
				print(label, path)
				if not label in label_ids:
					label_ids[label] = current_id
					current_id += 1
				id_ = label_ids[label]
				pil_image = Image.open(path).convert("L")
				size = (550, 550)
				final_image = pil_image.resize(size, Image.ANTIALIAS)
				image_array = np.array(final_image, "uint8")
				faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
				for (x,y,w,h) in faces:
					roi = image_array[y:y+h, x:x+w]
					x_train.append(roi)
					y_labels.append(id_)
	with open("pickles/face-labels.pickle", 'wb') as f:
		pickle.dump(label_ids, f)
	recognizer.train(x_train, np.array(y_labels))
	recognizer.save("recognizers/face-trainner.yml")
	recognizer = cv2.face.LBPHFaceRecognizer_create()
	recognizer.read("./recognizers/face-trainner.yml")
	return True

arguments = len(sys.argv) - 1
if arguments > 0:
	if sys.argv[1] == 'start':
		recognition()
	elif sys.argv[1] == 'train':
		train()
	else:
		exit()