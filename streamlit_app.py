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
    st.image("https://cdn.discordapp.com/attachments/961069393234767882/1240567424408092694/illustration-showing-diabetic-retinopathy.png?ex=66485972&is=664707f2&hm=bb69f02c1c56173d950d5505a98fa2bd7c8174b47a537e6aa276cadcd746e6da&"
          , use_column_width = True)
    st.markdown("""
          - Diabetic retinopathy is an eye condition of people with diabetes that can cause blindness or vision impairment [1]. 
          The prevalence of diabetic retinopathy is high among diabetic Filipino patients. 
          A study mentioned that for every 5 diabetic Filipino patient, 
          there is 1 patient with a sign of diabetic retinopathy[2]. As of now, there is no cure for diabetic retinopathy, but the sooner it is detected, the easier to treat or prevent the spread of it[3].

          
          """)

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
    st.markdown("## Aim of the :red[Study]")
    st.image("https://cdn.discordapp.com/attachments/961069393234767882/1240975794377986089/cataracta_consultation_lirema.png?ex=66488445&is=664732c5&hm=bc09a1bbce8e260269fc92226422b09a181d3aa12c8132ad7cabb90503fcf390&")
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

    image1DR = "https://cdn.discordapp.com/attachments/961069393234767882/1241003740035813406/00cb6555d108_png.rf.29cca170969c6e9918ef9b9209abef8e.jpg?ex=66489e4c&is=66474ccc&hm=03174074693ad13cc1ed360d9e62e1eeecbf1a361b74257dca2a97353b3ab406&"
    image2NODR = "https://cdn.discordapp.com/attachments/961069393234767882/1241003897657753730/0ae2dd2e09ea_png.rf.a4faf61bd46dc2930c51b3db7dba12cd.jpg?ex=66489e71&is=66474cf1&hm=6865068c3033afb596cf34fdadc9a7014d14c67644965a2881689fd3c823069d&"
    image3DR = "https://cdn.discordapp.com/attachments/961069393234767882/1241004684315983975/0f495d87656a_png.rf.707a2bb8a1223a714fcb88d67eb153c0.jpg?ex=66489f2d&is=66474dad&hm=12b8b9db28f22f7adfe6c320b7dc8c59b4cc8355adb570d25e246aa8aededf19&"
    image4NODR = "https://cdn.discordapp.com/attachments/961069393234767882/1241004929326252042/0daddc45d832_png.rf.8ebf5e03827f9b859246c465ffcfe7f3.jpg?ex=66489f67&is=66474de7&hm=42275fe31e738c3811d7e5d7033b3bfaa10a35f7eb94884527c64a8cd8391fb3&"

    list1 = ["Diabetic Retinopathy","Normal Retina","Diabetic Retinopathy"]

    st.image([image1DR, image2NODR, image3DR], use_column_width=False, caption = list1 )

    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)

    st.header("Trained :red[Model]" )

    code1 = '''final_model4 = Sequential([
                    Conv2D(8, (3, 3), activation='relu', input_shape=(150, 150, 3)),
                    MaxPooling2D((2, 2)),
                    Conv2D(16, (3, 3), activation='relu'),
                    MaxPooling2D((2, 2)),
                    Flatten(),
                    Dense(32, activation='relu'),
                    Dropout(0.2),
                    Dense(16, activation='relu'),
                    Dropout(0.2),
                    Dense(1, activation='sigmoid')
                ])

                final_model4.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])'''
    st.code(code1, language = 'python')

    st.markdown(""" - The model that we used is the code above, we used CNN because it is best suited for images. This is a model with 90.48% accuracy and 0.23 loss.
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
        st.image(image, use_column_width=True)
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
            st.image("https://cdn.discordapp.com/attachments/961069393234767882/1240936754912165918/image.png?ex=66485fe9&is=66470e69&hm=ee30e298caf453fa6d4e6d834e4029a70696b89bb0f691dbbcdb07788028ccfa&")

        with colr:
            st.write(' ')
        
        st.markdown("<p style=text-align: center; color: red; style=font-size:10%;><small>Hello, I am Christian Jay L. Cuevas,a 3rd year BS Computer Engineering student. I am in charge of preprocessing the images, training and saving the model. We collaborated in integrating the model and the streamlit. I hope that my knowledge can help other people, especially people with Diabetic Retinopathy.<small></p>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h3 style='text-align: center; color: white; ;'>Sean Julian Nicolas</h3>", unsafe_allow_html=True)
        coll, colm, colr = st.columns(3)

        with coll:
            st.write(' ')

        with colm:
            st.image("https://cdn.discordapp.com/attachments/961069393234767882/1240936615627718716/image.png?ex=66485fc8&is=66470e48&hm=22887278886e5fb7ff4fa27975db13f204917b0295d3f0e937bbd013c1e8a0fd&")

        with colr:
            st.write(' ')

        st.markdown("<p style=text-align: center; color: red; style=font-size:10%;><small>Hello, I am Sean Julian Segovia Nicolas, a 3rd year BS Computer Engineering student.  I am in charge of deploying the model and designing the streamlit. I also helped in finalizing the trained model. I hope to expand my knwoledge and create meaningful project that can help in the field of technology.<small></p>", unsafe_allow_html=True)
    st.markdown("""<hr style="height:10px;border:none;color:#EC4646;background-color:#EC4646;" /> """, unsafe_allow_html=True)
