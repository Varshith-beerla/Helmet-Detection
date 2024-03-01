import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
import os

model = YOLO("E:\\Helmet-Detection-YOLOv8-main\\models\\best.pt")
st.markdown('''<div style='text-align:center; font-size:50px; padding-bottom:2rem;'>Helmet Detection</div>''', unsafe_allow_html = True)

def remove():
  dir_path = 'E:\\Helmet-Detection-YOLOv8-main\\tests'
  for i in os.listdir(dir_path):
    path = os.path.join(dir_path, i)
    os.remove(path)

def predict(image_path, dir_path):
  image = cv2.imread(image_path)
  model.predict(image ,save =True,save_txt = True,conf=0.55,iou=0.6)
  predict_path = os.path.join(dir_path , 'predict')
  
  with open(os.path.join(predict_path , 'labels', 'image0.txt'), 'r') as file:
    label_txt = file.readlines()
    
  no_helmet_labels = []
  label_txt = [i.split(' ') for i in label_txt]
  for label in label_txt:
    if label[0] == "1":
      no_helmet_labels.append(label[1:])
  
  for label in no_helmet_labels:
    x,y,w,h = list(map(float, label))
    h_org, w_org, _ = image.shape
    abs_x = int((x - w/2) * w_org)
    abs_y = int((y - h/2) * h_org)
    abs_w = int(w * w_org)
    abs_h = int(h * h_org)
    cropped_image = image[abs_y : abs_y + abs_h, abs_x : abs_x + abs_w]
    cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    st.image(cropped_image_rgb, width=150)
  
  # predicted_image = cv2.imread(os.path.join(predict_path, 'image0.jpg'))
  # st.image(predicted_image, width=250)
  # os.remove('E:\\Helmet-Detection-YOLOv8-main\\runs\\detect\\predict\\labels\\image0.txt')
  # os.remove('E:\\Helmet-Detection-YOLOv8-main\\runs\\detect\\predict\\labels')
  # os.remove('E:\\Helmet-Detection-YOLOv8-main\\runs\\detect\\predict\\image0.jpg')
  # os.remove('E:\\Helmet-Detection-YOLOv8-main\\runs\\detect\\predict')

def getImage():
  image = st.file_uploader('Upload an Helmet Image', type = ['jpg', 'png', 'jpeg'])
  
  if image is not None:
    st.image(image, width=350)
    
    save_path = 'E:\\Helmet-Detection-YOLOv8-main\\tests' + f'\\{image.name}'
  
    if(st.button('Predict!')):
      with open(save_path, 'wb') as file:
        file.write(image.getbuffer())
      predict(save_path, 'E:\\Helmet-Detection-YOLOv8-main\\runs\\detect')
      
    if(st.button('Delete History!')):
      remove()
      
getImage()
    

# image=cv2.imread(r"E:\BikesHelmets143_png.rf.131436cd34ad67ebf54511e8251f35e1.jpg")
# plt.imshow(image)
# plt.axis('off')
# #print(np.array(x[0].boxes.data))

# labels=np.array(x[0].boxes.data)
# lis=[]
# for label in labels:
#     if label[5] == 1:
#         lis.append(label[:4])
# print(lis[0])

# if len(lis) == 1:
#   x,y,w,h = map(int,lis[0])
#   cropped_image = image[y : y + h, x : x +w]
#   cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
#   plt.imshow(cropped_image_rgb)
#   plt.axis('off')
# else:
#   fig, ax = plt.subplots(nrows=len(lis), figsize=(20,20 ))
#   for i in range(len(lis)):
    



