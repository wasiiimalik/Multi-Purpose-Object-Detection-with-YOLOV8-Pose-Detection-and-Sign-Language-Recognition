import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

st.set_page_config(page_title="YOLO Object Detection",
                   layout='wide',
                   page_icon='./images/object.png')

st.header('Get Object Detection for any Image')
st.write('Please Upload Image to get detections')

with st.spinner('Please wait while your model is loading'):
    yolo =  YOLO('/home/wmajidmalik/streamlineprojectforyolo_ActualModel/YoloV8WebProject/Model_Data/yolov8n.pt')

def upload_image():
    # Upload Image
    image_file = st.file_uploader(label='Upload Image')
    if image_file is not None:
        size_mb = image_file.size/(1024**2)
        file_details = {"filename":image_file.name,
                        "filetype":image_file.type,
                        "filesize": "{:,.2f} MB".format(size_mb)}
        #st.json(file_details)
        # validate file
        if file_details['filetype'] in ('image/png','image/jpeg'):
            st.success('VALID IMAGE file type (png or jpeg')
            return {"file":image_file,
                    "details":file_details}
        
        else:
            st.error('INVALID Image file type')
            st.error('Upload only png,jpg, jpeg')
            return None
        
def main():
    object = upload_image()
    
    if object:
        prediction = False
        image_obj = Image.open(object['file'])       
        
        col1 , col2 = st.columns(2)
        
        with col1:
            st.info('Preview of Image')
            st.image(image_obj)
            
        with col2:
            st.subheader('Check below for file details')
            st.json(object['details'])
            button = st.button('Get Detection from YOLO')
            if button:
                with st.spinner("""
                Geting Objets from image. please wait
                                """):
                    # below command will convert
                    # obj to array
                    #image_array = np.array(image_obj)
                    pred_img = yolo(image_obj)
                    res_plotted = pred_img[0].plot().astype('uint8')
                    pred_img_obj = Image.fromarray(res_plotted)
                    prediction = True
                
        if prediction:
            st.subheader("Predicted Image")
            st.caption("Object detection from YOLO V5 model")
            st.image(pred_img_obj)
    
    
    
if __name__ == "__main__":
    main()