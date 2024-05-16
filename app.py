import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns

# Set the page config
st.set_page_config(page_title="University Recommendation System", page_icon="🎓", layout="wide")

# Load the expanded dataset
@st.cache_data
def load_data():
    return pd.read_csv("university_data.csv")

df = load_data()

# Load the trained model and preprocessors
@st.cache_resource
def load_model_and_preprocessors():
    model = load_model('university_recommendation_model.h5')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('label_encoder_course.pkl', 'rb') as f:
        label_encoder_course = pickle.load(f)
    with open('label_encoder_uni.pkl', 'rb') as f:
        label_encoder_uni = pickle.load(f)
    return model, scaler, label_encoder_course, label_encoder_uni

model, scaler, label_encoder_course, label_encoder_uni = load_model_and_preprocessors()

# Function to generate personalized advice without OpenAI
def generate_personalized_advice(university, course, marks):
    advice = []

    if marks['Science'] < 50:
        advice.append("Focus on improving your Science marks.")
    if marks['Maths'] < 50:
        advice.append("Focus on improving your Maths marks.")
    if marks['History'] < 50:
        advice.append("Focus on improving your History marks.")
    if marks['English'] < 50:
        advice.append("Focus on improving your English marks.")
    if marks['GRE'] < 300:
        advice.append("Consider retaking the GRE to improve your score.")

    if course == "Engineering":
        advice.append("Enhance your programming skills by taking online courses.")
    elif course == "Computer Science":
        advice.append("Work on projects and internships related to software development.")
    elif course == "Physics":
        advice.append("Participate in research projects and science fairs.")
    elif course == "History":
        advice.append("Read extensively and engage in historical research projects.")
    elif course == "Chemistry":
        advice.append("Gain hands-on experience in laboratories and participate in chemistry competitions.")
    elif course == "MBBS":
        advice.append("Gain practical experience by volunteering at clinics or hospitals.")
    elif course == "Literature":
        advice.append("Engage in extensive reading and writing practice.")
    elif course == "Business Administration":
        advice.append("Develop leadership and management skills through relevant courses and activities.")

    return advice

# Function to predict the university, recommended course, and advice based on input marks
def predict_university_and_advice(science_marks, maths_marks, history_marks, english_marks, gre_marks):
    try:
        input_data = np.array([[science_marks, maths_marks, history_marks, english_marks, gre_marks]])
        input_data = scaler.transform(input_data)
        prediction = model.predict(input_data)
        predicted_uni_index = np.argmax(prediction)
        predicted_uni = label_encoder_uni.inverse_transform([predicted_uni_index])[0]

        if predicted_uni in df['University Name'].values:
            university_info = df[df['University Name'] == predicted_uni].iloc[0]
            university_link = university_info['University Link']
            scholarship_info = university_info['Scholarship Info']
            academic_fee = university_info['Academic Fee']
            recommended_course = university_info['Course']

            marks = {
                "Science": science_marks,
                "Maths": maths_marks,
                "History": history_marks,
                "English": english_marks,
                "GRE": gre_marks
            }
            personalized_advice = generate_personalized_advice(predicted_uni, recommended_course, marks)

            return predicted_uni, university_link, scholarship_info, academic_fee, recommended_course, personalized_advice
        else:
            st.error(f"Predicted university '{predicted_uni}' not found in the dataset.")
            return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return None, None, None, None, None, None

st.title('University Recommendation System')
st.subheader('Advisor: Dr. Neha Chauhan')

st.markdown("""
Welcome to the University Recommendation System! Input your academic marks and GRE score to receive a personalized university and course recommendation along with some advice to help you succeed.
""")

# Sidebar for user inputs
st.sidebar.header("Enter Your Marks")

science_marks = st.sidebar.slider('Science Marks', min_value=0.0, max_value=100.0, value=75.0)
maths_marks = st.sidebar.slider('Maths Marks', min_value=0.0, max_value=100.0, value=75.0)
history_marks = st.sidebar.slider('History Marks', min_value=0.0, max_value=100.0, value=75.0)
english_marks = st.sidebar.slider('English Marks', min_value=0.0, max_value=100.0, value=75.0)
gre_marks = st.sidebar.slider('GRE Marks', min_value=0.0, max_value=340.0, value=300.0)

if st.sidebar.button('Submit'):
    university, link, scholarship, fee, course, advice = predict_university_and_advice(science_marks, maths_marks, history_marks, english_marks, gre_marks)
    
    if university:
        st.write(f"### Recommended University: **[{university}]({link})**")
        st.write(f"### Recommended Course: **{course}**")
        st.write(f"### Scholarship Information: [Link]({scholarship})")
        st.write(f"### Academic Fee: **{fee}**")
        
        st.write("### Personal Advice:")
        for item in advice:
            st.write(f"- {item}")
        
        st.write("### Your Marks Overview")
        marks = {
            'Subjects': ['Science', 'Maths', 'History', 'English', 'GRE'],
            'Marks': [science_marks, maths_marks, history_marks, english_marks, gre_marks]
        }
        marks_df = pd.DataFrame(marks)
        
        fig, ax = plt.subplots()
        sns.barplot(x='Subjects', y='Marks', data=marks_df, ax=ax)
        ax.set_ylim(0, 100)
        st.pyplot(fig)

st.sidebar.markdown("""
---
### About
This app provides university and course recommendations based on your academic marks and GRE score, along with personalized advice to help you achieve your goals.
""")

st.image("top-10-universities-in-the-world.png", caption="Achieve Your Academic Goals!", use_column_width=True)




