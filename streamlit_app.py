import streamlit as st
from streamlit_option_menu import option_menu
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

st.set_page_config(
    page_title= "Retinopathy",
    page_icon = "👁️"
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Deep Learning Model", "About Us"],
    icons = ["house","boxes", "chat"],
    default_index=0,
    orientation="horizontal",
    styles={
        "icon": {"font-size": "18px"},
        "nav-link": {
            "font-size": "15px",
            "text-align": "center",
        }
    }
)

if selected == "Home":
    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
    st.markdown("## What is :red[Diabetic Retinopathy?]")
    st.image("IMG1HOME.png"
          , use_container_width = True)
    
    st.markdown("""
          - Diabetic retinopathy is an eye condition of people with diabetes that can cause blindness or vision impairment [1]. 
          The prevalence of diabetic retinopathy is high among diabetic Filipino patients. 
          A study mentioned that for every 5 diabetic Filipino patient, 
          there is 1 patient with a sign of diabetic retinopathy[2]. As of now, there is no cure for diabetic retinopathy, but the sooner it is detected, the easier to treat or prevent the spread of it[3].

          
          """)

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
    st.markdown("## Aim of the :red[Study]")
    st.image("IMG2HOME.png")
    
    st.markdown("""- Early detection of diabetic retinopathy is essential to its treatment[3]. The diagnosis of the diabetic retinopathy is done by
          capturing images of your retina using special cameras[4]. Then the assessment relies solely on the manual interpretation of the doctor,
          which can lead to human-error or can be time consuming.""")
          
    st.markdown("""- The goal of this study is to create a model using CNN that can help with the early detection if the patient has diabetic retinopathy.
          This can help reduce time and significantly reduce any errors that may occur. We are using CNN because it can achieve state-of-the-art
          results and it is simple to use[5]
          """)

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)

    st.header("About the :red[Dataset]")
    st.markdown(""" - The dataset that we have chosen is the diagnosis of diabetic retinopathy dataset from kaggle. This is a dataset that contains images of
            retina with and without Diabetic Retinopathy. The original dataset has a large amount of images, we just decided to get a part of it which are
            the images of retina without DR and with DR.  """)

    image1DR = "IMG3HOMEDR.jpg"
    image2NODR = "IMG4HOMENODR.jpg"
    image3DR = "IMG5HOMEDR.jpg"

    list1 = ["Diabetic Retinopathy","Normal Retina","Diabetic Retinopathy"]

    st.image([image1DR, image2NODR, image3DR], use_container_width=False, caption = list1 )

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)

    st.header("Trained :red[Model]" )

    code1 = '''final_model4 = Sequential([
                    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                    MaxPooling2D((2, 2)),
                    Conv2D(32, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.2),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])  
                final_model4.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])'''
    st.code(code1, language = 'python')

    st.markdown(""" - The model that we used is the code above, we used CNN because it is best suited for images. This is a model with 93.07% accuracy and 0.23 loss.
                  It can be further improved by using a larger dataset and doing more image augmentation but since we have a limitation in the size of the model
                  we chose to keep it small.""")

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)

    st.header("References")

    st.markdown(""" [1]National Eye Institute, “Diabetic Retinopathy | National Eye Institute,” Nih.gov, 2019. https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy#:~:text=Diabetic%20retinopathy%20is%20an%20eye  
                [2]PJOteam, “Prevalence of diabetic retinopathy among diabetic patients in a tertiary hospital,” Philippine Journal Of Ophthalmology, Feb. 07, 2019. https://paojournal.com/article/prevalence-of-diabetic-retinopathy-among-diabetic-patients-in-a-tertiary-hospital/  
                [3]“Treatments,” stanfordhealthcare.org. https://stanfordhealthcare.org/medical-conditions/eyes-and-vision/diabetic-retinopathy/treatments.html#:~:text=There%20is%20no%20cure%20for  
                [4]K. Boyd, “What Is Diabetic Retinopathy?,” American Academy of Ophthalmology, Oct. 24, 2019. https://www.aao.org/eye-health/diseases/what-is-diabetic-retinopathy  
                [5]J. Brownlee, “Crash Course in Convolutional Neural Networks for Machine Learning,” Machine Learning Mastery, Jun. 23, 2016. https://machinelearningmastery.com/crash-course-convolutional-neural-networks/
                """)

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
    
if selected == "Deep Learning Model":
    @st.cache(allow_output_mutation=True)
    def load_model():
        try:
            model = tf.keras.models.load_model('model3 (1).h5')
            string2 = "Model loaded successfully."
            st.success(string2)
            return model
        except Exception as e:
            st.error("Error loading the model. Please check if the model file exists and is valid.")
            st.error(str(e))
            return None

    model = load_model()

    st.markdown("""
            ## Diabetic Retinopathy :red[Detection System]"""
            )

    file = st.file_uploader("Choose retina image from computer", type=["jpg", "png"])

    def import_and_predict(image_data, model):
        size = (150, 150)
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis, ...]
        prediction = model.predict(img_reshape)
        return prediction

    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file).convert("RGB")
        st.image(image, use_container_width=True)
        prediction = import_and_predict(image, model)
        if prediction is not None:
            class_names = ['Normal Retina', 'Diabetic Retinopathy']
            if prediction > 0.5:
                string = "OUTPUT : " + class_names[0]
            else:
                string = "OUTPUT : " + class_names[1]
            st.success(string)

if selected == "About Us":

    st.markdown("# About:red[ Us]")
    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
    col1, col2 = st.columns(2, gap="medium")

    with col1:
        st.markdown("<h3 style='text-align: center; color: white; ;'>Christian Jay Cuevas</h3>", unsafe_allow_html=True)
        coll, colm, colr = st.columns(3)

        with coll:
            st.write(' ')

        with colm:
            st.image("IMG7JAY.png")

        with colr:
            st.write(' ')
        
        st.markdown("<p style=text-align: center; color: red; style=font-size:10%;><small>Hello, I am Christian Jay L. Cuevas,a 3rd year BS Computer Engineering student. I am in charge of preprocessing the images, training and saving the model. We collaborated in integrating the model and the streamlit. I hope that my knowledge can help other people, especially people with Diabetic Retinopathy.<small></p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: center; color: white; ;'>Sean Julian Nicolas</h3>", unsafe_allow_html=True)
        coll, colm, colr = st.columns(3)

        with coll:
            st.write(' ')

        with colm:
            st.image("IMG6SHAN.png")

        with colr:
            st.write(' ')

        st.markdown("<p style=text-align: center; color: red; style=font-size:10%;><small>Hello, I am Sean Julian S. Nicolas, a 3rd year BS Computer Engineering student.  I am in charge of deploying the model and designing the streamlit. I also helped in finalizing the trained model. I hope to expand my knowledge and create meaningful project that can help in the field of technology.<small></p>", unsafe_allow_html=True)
    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
