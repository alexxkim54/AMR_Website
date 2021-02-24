import streamlit as st
import numpy as np
import pandas as pd

st.title('Makeup Recommender')

st.header('About the Recommender')
st.write('This webpage is a recommender that will provide a one-stop shop experience where a user will get recommended an array of products to create an entire makeup look based on similar products that the user enjoys, products that similar users have purchased, as well as products that are personalized to the user including skin type, skin tone, allergies, and budget. Our project aims to utilize collaborative filtering recommendations along with content-based filtering to ensure user satisfaction and success when creating their desired look.')
st.subheader('First, we need to ask you some questions...')
skin_type = st.radio('Which best describes your skin type?', ('Oily', 'Dry', 'Combination', 'Normal'))
skin_tone = st.radio('Which best describes your skin tone?', ('Porcelain', 'Fair', 'Light', 'Medium', 'Tan', 'Olive', 'Deep', 'Dark', 'Ebony'))
budget = st.slider('What is your budget per product?', 0, 240, 0, 5)
allergies = st.multiselect('Do you wish to avoid any of the following ingredients? Select those that apply.', ['Rubber', 'Fragrances', 'Preservatives', 'Metals'])
products = st.text_input('We would love to know which products you currently enjoy! Please enter at least one Sephora product ID below (separated by commas).')
st.button('Find me a set!')

st.sidebar.title('Contact Us')
st.sidebar.text('Name, Emails')
st.sidebar.title('Repo')
st.sidebar.text('https://github.com/alexxkim54/AMR_Website')