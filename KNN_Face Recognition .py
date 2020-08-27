import cv2
import numpy as np 
import os





face_cascade = cv2.CascadeClassifier("C:\\Users\\OmerF\\anaconda3\\Lib\\site-packages\\cv2\data\\haarcascade_frontalface_default.xml")

dataset_path = "C:\\Users\\OmerF\\.spyder-py3\\fotolar\\"


def veri_olustur():
    cap = cv2.VideoCapture(0)
    skip = 0 
    face_data = [] 
    
    file_name = input("Enter the name of person : ")
    
    
    while True:
    	ret,frame = cap.read()
    
    	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 
    
    	if ret == False: 
    		continue
    
    	
    	faces = face_cascade.detectMultiScale(gray_frame,1.3,5) 
    	if len(faces) == 0:
    		continue
    
    	k = 1
    
    	
    	faces = sorted(faces, key = lambda x : x[2]*x[3] , reverse = True) 
    
    	
    	skip += 1
    
    	
    	for face in faces[:1]: 
    		x,y,w,h = face
    
    		offset = 5 
    		face_offset = frame[y-offset:y+h+offset,x-offset:x+w+offset]
    		face_selection = cv2.resize(face_offset,(100,100)) 
    
    		
    		if skip % 10 == 0:
    			face_data.append(face_selection)
    			print (len(face_data))
    
    
    		cv2.imshow(str(k), face_selection)
    		k += 1
    		
    		
    		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2) 
    		
    
    	cv2.imshow("faces",frame)
    
    	
    	key_pressed = cv2.waitKey(1) & 0xFF 
    	if key_pressed == ord('q'): 
    		break
    cap.release()
    cv2.destroyAllWindows()
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))
    print (face_data.shape)
    
    
    np.save(dataset_path + file_name, face_data)
    print ("Dataset saved at : {}".format(dataset_path + file_name + '.npy'))
    
    






def egitim():
    cap = cv2.VideoCapture(0) 
    def distance(v1, v2): 
    	return np.sqrt(((v1-v2)**2).sum())
    
    def knn(train, test, k=5):
    	dist = []
    	
    	for i in range(train.shape[0]):
    		ix = train[i, :-1]
    		iy = train[i, -1]
    		d = distance(test, ix)
    		dist.append([d, iy])
    	dk = sorted(dist, key=lambda x: x[0])[:k]
    	labels = np.array(dk)[:, -1]
    	
    	
    	output = np.unique(labels, return_counts=True)
    	index = np.argmax(output[1])
    	return output[0][index]

    
    dataset_path = "C:\\Users\\OmerF\\.spyder-py3\\fotolar\\"
    
    face_data = [] 
    labels = [] 
    class_id = 0 
    names = {}
    
    

    for fx in os.listdir(dataset_path): 
    	if fx.endswith('.npy'):
    		names[class_id] = fx[:-4]
    		data_item = np.load(dataset_path + fx) 
    		face_data.append(data_item) 
    
    		
    		target = class_id * np.ones((data_item.shape[0],))
    		class_id += 1
    		labels.append(target)
    
    face_dataset = np.concatenate(face_data, axis=0) 
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    print(face_labels.shape)
    print(face_dataset.shape)
    
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    print(trainset.shape)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    
    while True:
    	ret, frame = cap.read()
    	if ret == False:
    		continue
    	
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	
    	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    	for face in faces:
    		x, y, w, h = face
    
    		
    		offset = 5
    		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
    		face_section = cv2.resize(face_section, (100, 100))
    
    		out = knn(trainset, face_section.flatten())
    
    		
    		cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
    
    	cv2.imshow("Faces", frame)
    
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
    
    cap.release()
    cv2.destroyAllWindows()
        






    
while True:
    dataset_path = "C:\\Users\\OmerF\\.spyder-py3\\fotolar\\"
    sayi=0
    for fx in os.listdir(dataset_path):
    	if fx.endswith('.npy'):
            sayi +=1
    print(sayi)        
    if sayi==0:
        veri_olustur()
        egitim()
   
    cap = cv2.VideoCapture(0) 
   
    def distance(v1, v2):
    	# Eucledian 
    	return np.sqrt(((v1-v2)**2).sum())
    
    def knn(train, test, k=5):
    	dist = []
    	
    	for i in range(train.shape[0]):
    		
    		ix = train[i, :-1]
    		iy = train[i, -1]
    		d = distance(test, ix)
    		dist.append([d, iy])
    	dk = sorted(dist, key=lambda x: x[0])[:k]
    	labels = np.array(dk)[:, -1]
    	
    	
    	output = np.unique(labels, return_counts=True)
    	index = np.argmax(output[1])
    	return output[0][index]
 
    
 
    
    dataset_path = "C:\\Users\\OmerF\\.spyder-py3\\fotolar\\"
    
    face_data = [] 
    labels = []
    class_id = 0
    names = {} 

    
    
    for fx in os.listdir(dataset_path): 
    	if fx.endswith('.npy'):
    		names[class_id] = fx[:-4]
    		data_item = np.load(dataset_path + fx) 
    		face_data.append(data_item) 
    		target = class_id * np.ones((data_item.shape[0],))
    		class_id += 1
    		labels.append(target)
    
    face_dataset = np.concatenate(face_data, axis=0) 
    face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
    print(face_labels.shape)
    print(face_dataset.shape)
    
    trainset = np.concatenate((face_dataset, face_labels), axis=1)
    print(trainset.shape)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    

    
    while True:
    	ret, frame = cap.read() 
    	if ret == False:
    		continue
    	
    	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    	
    	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    	for face in faces:
    		x, y, w, h = face
    
    		
    		offset = 5
    		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
    		face_section = cv2.resize(face_section, (100, 100))
    
    		out = knn(trainset, face_section.flatten())
            
    		cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
    		cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            
                
    	cv2.imshow("Faces", frame)
    
    	if cv2.waitKey(1) & 0xFF == ord('q'):
    		break
    
    cap.release()
    cv2.destroyAllWindows()
        