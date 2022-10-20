from torchvision import models
import torch
import streamlit as st
from PIL import Image
import numpy as np
# import cv2

alexnet = models.alexnet(weights='IMAGENET1K_V1')


from torchvision import transforms
transform = transforms.Compose([            #[1]
 transforms.Resize(256),                    #[2]
 transforms.CenterCrop(224),                #[3]
 transforms.ToTensor(),                     #[4]
 transforms.Normalize(                      #[5]
 mean=[0.485, 0.456, 0.406],                #[6]
 std=[0.229, 0.224, 0.225]                  #[7]
 )])

# read labels if images
with open('./data/imagenet1000_clsidx_to_labels.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return im, image

# Uploading the File to the Page
uploadFile = st.file_uploader(label="Upload image", type=['jpg', 'png'])

if uploadFile is not None:
    img, image = load_image(uploadFile)
    st.image(img)
    st.write("Image Uploaded Successfully")

    img_t = transform(img)
    # img_t = transform(opencv_image)
    batch_t = torch.unsqueeze(img_t, 0)

    alexnet.eval()

    out = alexnet(batch_t)
    
    _, index = torch.max(out, 1) 
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    print(labels[index[0]], percentage[index[0]].item())

    # print(f'type(labels[index[0]]): {type(labels[index[0]])}')

    st.title('Prediction of the AlexNet')

    st.metric(label="Perhaps its a", value=labels[index[0]], delta=percentage[index[0]].item())


