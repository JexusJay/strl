import streamlit as st
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans

# Load the trained models
with open('model.pkl', 'rb') as file:
    nb = pickle.load(file)

def predict_naive_bayes(q1, q2, q3, q4, q5, q6, q7, q8):
    # Create a pandas DataFrame with the input values
    input_data = pd.DataFrame({'q1': [q1], 'q2': [q2], 'q3': [q3], 'q4': [q4], 'q5': [q5], 'q6': [q6], 'q7': [q7], 'q8': [q8]})
    
    # Define a mapping from string values to integer values
    mapping = {'Yes': 2, 'No': 0, 'Sometimes': 1}
    
    # Apply the mapping to the input values
    input_data = input_data.applymap(lambda x: mapping.get(x, x))
    
    # Make a prediction using the Naive Bayes algorithm
    prediction = nb.predict(input_data)
    
    # Map the integer prediction to the corresponding word
    if prediction[0] == 0:
        prediction_word = 'less addictive to alcohol'
    elif prediction[0] == 1:
        prediction_word = 'mildly addictive to alcohol'
    else:
        prediction_word = 'high addiction to alcohol'
    
    # Return the predicted value
    return prediction_word

# Create a Streamlit app
st.title('Alcohol addiction level classifier')

st.subheader('Have you experienced High blood preasure this past few months?')
q1 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=1, horizontal=True)

st.subheader('How often have you been stressed this past few months/week?')
q2 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=2, horizontal=True)

st.subheader('Do you experience movement coordination problems?')
q3 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=3, horizontal=True)

st.subheader('Do you experience any speech problems lately?')
q4 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=4, horizontal=True)

st.subheader('Are you experiencing forgetfulness?')
q5 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=5, horizontal=True)

st.subheader('Have you experienced eyesight or vision problems?')
q6 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=6, horizontal=True)

st.subheader('Do you experience any breathing problems?')
q7 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=7, horizontal=True)

st.subheader('Have you experience nausea?')
q8 = st.radio('', ["Yes", "No", "Sometimes"], index=0, key=8, horizontal=True)

if st.button('Classify'):
    result = predict_naive_bayes(q1, q2, q3, q4, q5, q6, q7, q8)
    st.write("The predicted alcohol addiction level is:", result)

