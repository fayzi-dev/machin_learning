import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('mymodel.joblib')

# Title and description
st.title('پیش بینی حقوق برنامه نویسان در سال 2022')
st.write('لطفاْ فیلد های زیر را برای پیش بینی دقیق تکمیل نمایید.')

# Country selection
countries = (
    'United States of America', 'Germany', 'United Kingdom of Great Britain and Northern Ireland', 
    'India', 'Canada', 'France', 'Brazil', 'Spain', 'Poland', 'Netherlands', 'Australia', 
    'Italy', 'Sweden', 'Russian Federation', 'Switzerland', 'Turkey', 'Austria', 'Israel', 
    'Czech Republic', 'Belgium', 'Portugal', 'Denmark', 'Mexico', 'Norway', 'Romania', 
    'Greece', 'Pakistan', 'New Zealand', 'Finland', 'Argentina', 'South Africa', 
    'Iran, Islamic Republic of...', 'Ukraine', 'Hungary', 'Bangladesh', 'Ireland', 'Japan', 
    'Colombia', 'Other'
)
country = st.selectbox('انتخاب کشور ', countries)

# Education level selection
education_levels = [
    'Less than a Bachelors',
    'Bachelor’s degree',
    'Master’s degree'
]
education = st.radio('سطح تحصیلات', education_levels)

# Experience selection
experience = st.slider('(سال)سابقه کار', 0, 50, 2)

# Predict button
ok = st.button('محاسبه حقوق')

# Salary prediction
if ok:
    # Prepare input data as a DataFrame directly
    X_new_df = pd.DataFrame([[country, education, experience]], columns=['Country', 'EdLevel', 'YearsCodePro'])
    
    # Predict salary
    salary = model.predict(X_new_df)
    
    # Display predicted salary
    st.subheader(f'حقوق پیش بینی شده: ${salary[0]:,.2f}')


# 2. Example code 2


# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib


# model  = joblib.load('mymodel.joblib')
# st.title('پیش بینی حقوق برنامه نویسان در سال 2022')
# st.write('لطفاْ فیلد های  زیر را برای  پیش بینی دقیق تکمیل نمایید .')

# countries = (
#     'United States of America',
#     'Germany',
#     'United Kingdom of Great Britain and Northern Ireland',
#     'India',
#     'Canada',
#     'France',
#     'Brazil',
#     'Spain',
#     'Poland',
#     'Netherlands',
#     'Australia',
#     'Italy',
#     'Sweden',
#     'Russian Federation',
#     'Switzerland',
#     'Turkey',
#     'Austria',
#     'Israel',
#     'Czech Republic',
#     'Belgium',
#     'Portugal',
#     'Denmark',
#     'Mexico',
#     'Norway',
#     'Romania',
#     'Greece',
#     'Pakistan',
#     'New Zealand',
#     'Finland',
#     'Argentina',
#     'South Africa',
#     'Iran, Islamic Republic of...',
#     'Ukraine',
#     'Hungary',
#     'Bangladesh',
#     'Ireland',
#     'Japan',
#     'Colombia',
#     'Other',
# )

# education = [
#     'Less than a Bachelors',
#     'Bachelor’s degree',
#     'Master’s degree'
# ]

# country = st.selectbox('انتخاب کشور ', countries)
# # education = st.selectbox('سطح تحصیلات', education)
# education = st.radio('سطح تحصیلات', education)
# expericence = st.slider('(سال)سابقه کار', 0, 50, 2)

# columns = ['Country', 'EdLevel', 'YearsCodePro']

# ok = st.button('محاسبه حقوق')

# if ok:
#     X_new = [countries,education,expericence]
#     X_new_df = pd.DataFrame([X_new], columns= columns)
#     salary = model.predict(X_new_df)

#     # Display predicted salary

#     # st.subheader(salary[0])
#     # st.subheader(f'حقوق پیش بینی شده: ${salary[0]:,.2f}')
#     st.subheader(f'حقوق پیش بینی شده بر اساس داده های ورودی ${salary[0]:.2f} می باشد.')
     


