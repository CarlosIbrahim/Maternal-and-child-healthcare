import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

import sqlite3
import hashlib
import joblib

#reading the data
Stillbirth= pd.read_csv('StillbirthRate.csv')
MaternalMortality= pd.read_csv('MaternalMortalityRatio.csv')
Under5Mortality = pd.read_csv('Under5MortalityRate.csv')

#hiding streamlit waterprint
hide_streamlit_style = """<style> footer {visibility: hidden;} </style> """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

#signin/signup - linking it to database with usernames and passwords
conn=sqlite3.connect('data.db')
c=conn.cursor()

def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username=? AND password=?',(username,password))
    data=c.fetchall()
    return data

def view_all_users():
    c.execute('SELECT * FROM userstable')
    data=c.fetchall()
    return data

def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def verify_hashes(password,hashed_text):
    if generate_hashes(password)==hashed_text:
        return hashed_text
    return false

def main():
    #loginpage
    st.title("Mother and Child Wellbeing\n Are SGDs on track?")
    #st.image('https://sdgs.un.org/sites/default/files/2020-07/The-Sustainable-Development-Goals-Report-2020_Page_10.png', width=900)
    menu=["Login","SignUp"]
    signuplogin=st.sidebar.selectbox("Sign up or Login here.",menu)
    if signuplogin=="Login":
        username=st.sidebar.text_input("Username")
        password=st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pass=generate_hashes(password)
            result=login_user(username,verify_hashes(password,hashed_pass))
            if result:
                st.sidebar.success("Logged in as {} ".format(username))

                #content menu
                contents = st.sidebar.selectbox('Analysis contents:',['Trends until 2019', 'Future Trends'])
                if contents == 'Trends until 2019':
                    select_year = st.slider('Year', 1950, 2019,[2000,2019])

                    #Stillbirth plot including country selection
                    subset_stillbirth = Stillbirth[Stillbirth['Year'].between(select_year[0],select_year[1])]
                    country_name_input = st.multiselect(
                    'Select Country/ies',
                    Stillbirth.groupby('Country').count().reset_index()['Country'].tolist())
                    if len(country_name_input) > 0:
                        subset_stillbirth = Stillbirth[Stillbirth['Country'].isin(country_name_input)]

                    fig01 = px.bar(subset_stillbirth, x="Year", y="Value", color='WHO Region', barmode="group",title="Stillbirth rate (per 1000 total births) in WHO Regions",width=800, height=400)
                    fig01.update_layout(
                            autosize=False,
                            yaxis=dict(
                            titlefont=dict(size=20))
                    )
                    st.plotly_chart(fig01)

                    #Children under 5 mortality plot following same country selection as previous plot
                    subset_under5mortality = Under5Mortality
                    subset_under5mortality = subset_under5mortality[subset_under5mortality['Year'].between(select_year[0],select_year[1])]
                    if len(country_name_input) > 0:
                        subset_under5mortality = subset_under5mortality[subset_under5mortality['Country'].isin(country_name_input)]
                    fig02 = px.bar(subset_under5mortality, x="Year", y="Value", color='WHO Region', barmode="group",title="Under-five mortality rate (per 1000 live births) in WHO Regions",width=800, height=400)
                    fig02.update_layout(
                            autosize=False,
                            yaxis=dict(
                            titlefont=dict(size=20))
                    )
                    st.plotly_chart(fig02)

                    #Maternal Mortality plot following same country selection as previous plot
                    subset_maternalmortality = MaternalMortality[MaternalMortality['Year'].between(select_year[0],select_year[1])]
                    if len(country_name_input) > 0:
                        subset_maternalmortality = MaternalMortality[MaternalMortality['Country'].isin(country_name_input)]
                    fig03 = px.bar(subset_maternalmortality, x="Year", y="Value", color='WHO Region', barmode="group",title="Maternal Mortality (per 1000 total births) in WHO Regions",width=800, height=400)
                    fig03.update_layout(
                            autosize=False,
                            yaxis=dict(
                            titlefont=dict(size=20))
                    )
                    st.plotly_chart(fig03)

                if contents =='Future Trends':
                    st.text("Linear Regression")

                    y1 =Stillbirth.Value
                    X1 =Stillbirth[["Year"]].values
                    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.3)
                    model1 = LinearRegression()
                    model1.fit(X1_train,y1_train)
                    pred1 = model1.predict(X1_test)
                    #plotting the regression
                    fig1 =plt.figure(figsize=(4,3))
                    #ax1 = plt.axes()
                    #ax1.scatter(X1, y1)
                    plt.plot(X1_test, pred1)
                    plt.legend()
                    plt.show()
                    st.plotly_chart(fig1)
                    #calculating model errors
                    mse1 = mean_squared_error(y1_test,pred1)
                    rmse1 = sqrt(mse1)
                    st.markdown(f"""
                    Linear Regression model trained on Stillbirth Rate data:
                    - MSE:{mse1}
                    - RMSE:{rmse1}
                    """)
                    st.success('Model for Stillbirth Rate trained successfully')

                    y2 =Under5Mortality.Value
                    X2 =Under5Mortality[["Year"]].values
                    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3)
                    model2 = LinearRegression()
                    model2.fit(X2_train,y2_train)
                    pred2 = model2.predict(X2_test)
                    mse2 = mean_squared_error(y2_test,pred2)
                    rmse2 = sqrt(mse2)
                    st.markdown(f"""
                    Linear Regression model trained on Stillbirth Rate data:
                    - MSE:{mse2}
                    - RMSE:{rmse2}
                    """)
                    st.success('Model for Under 5 Child Mortality Rate trained successfully')

                    y3 =Under5Mortality.Value
                    X3 =Under5Mortality[["Year"]].values
                    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3)
                    model3 = LinearRegression()
                    model3.fit(X3_train,y3_train)
                    pred3 = model3.predict(X3_test)
                    mse3 = mean_squared_error(y3_test,pred3)
                    rmse3 = sqrt(mse3)
                    st.markdown(f"""
                    Linear Regression model trained on Stillbirth Rate data:
                    - MSE:{mse3}
                    - RMSE:{rmse3}
                    """)
                    st.success('Model for Maternal Mortality Ratio trained successfully')



            else:
                st.warning("Incorrect Username or Password")
    elif signuplogin=="Sign Up":
        st.subheader("Create New Account")
        new_username=st.sidebar.text_input("Username")
        new_password=st.sidebar.text_input("Password",type='password')
        confirm_password=st.sidebar.text_input("Confirm Password",type='password')
        if new_password==confirm_password:
            st.success("Password confirmed.")
        else:
            st.warning("Passwords are not matching. Kindly retype them.")
        if st.sidebar.button("Signup"):
            create_usertable()
            hashed_new_password=generate_hashes(new_password)
            add_userdata(new_username,hashed_new_password)
            st.success("You have created a new account")
            st.info("Go To Login Page")
#log_home=st.sidebar.checkbox('Login To Start')
#if log_home:
main()
