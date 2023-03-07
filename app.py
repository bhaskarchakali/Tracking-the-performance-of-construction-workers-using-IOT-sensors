
import streamlit as st
import numpy as np
import pickle

emp_perf_model_path1 = open("constructionmodel.pkl","rb")
emp_perf_model1=pickle.load(emp_perf_model_path1)

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>',unsafe_allow_html = True)


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 



if emp_perf_model_path1:
    primary_clr = st.get_option("theme.primaryColor")
    txt_clr = st.get_option("theme.textColor")
    
    second_clr = "#d87c7c"
    primary_clr = '#4bff58'
    second_clr = '33332f'
    backgroundColor='#0e0e0e'
    txt_clr = '#f9f9fb'
else:
    primary_clr = '#4bff58'
    second_clr = '33332f'
    backgroundColor='#0e0e0e'
    txt_clr = '#f9f9fb'







def main():
    st.title(":white[Construction Worker's Performance Prediction]")
    
    a = st.slider("**Age(years)**",18,50)
    b = st.selectbox("**Gender**",('Female','Male'))
    if b == 'Female':
        b=0
    else:
        b=1
    c = st.selectbox("**Nationality**",("Bangladesh", "China","Japan","Malaysia","Pakistan","Philippines","Sri Lanka"))
    if c == "Bangladesh":
        c=0
    elif c == "China":
        c=1    
    elif c == "Japan":
        c=2
    elif c == "Malaysia":
        c=3
    elif c == "Pakistan":
        c=4
    elif c == "Philippines":
        c=5
    else:
        c=6
    d = st.selectbox("**Site**",('Site-1','Site-2','Site-3')) 
    if d == "Site-1":
         d=0
    elif d== "Site-2":
         d=1
    else:
         d=2 
    e = st.selectbox("**Working_Hours**",('Equal to 9hrs','Greater than 9hrs','Less than 9hrs'))
    if e == 'Equal to 9hrs':
    	e = 0
    elif e == 'Greater than 9hrs':
    	e = 1
    else:
        e = 2
    f = st.slider("**Experience(years)**",0,10)
    g = st.selectbox("**Attendance**",('Poor','Good'))
    if g == 'Poor':
       g = 0
    else:
        g = 1  
    h = st.selectbox("**Noise_Detection**",('Yes','No'))
    if h == 'No':
        h = 0
    else:
        h = 1      
    i = st.selectbox("**Infrared_Sensor**",('Yes','No'))
    if i == 'No':
        i = 0
    else:
        i = 1
    j = st.selectbox("**Gas_Sensor**",('Yes','No'))
    if j == 'No':
        j = 0
    else:
        j = 1   
    k = st.selectbox("**Galvanic_Skin_Response_Sensor**",('Yes','No'))
    if k == 'No':
        k = 0
    else:
        k = 1  
   # l = st.number_input('**Temperature**')
    m = st.slider("**Respiratory_Rate(RR)**",0,30)
    n = st.slider('**Heart_Rate(HR)**',80,160)
    o = st.selectbox("**Work_Position**",('Adjusting Bricks','Crushing','Curing','Cutting Bricks','Cutting Re-bar','Drilling','Excavation','Fetching Mortar','Lashing Rope','Moving Bricks','Moving Stones','Placing Rebar','Re-bar','Welding','Wiring'))
    if o == 'Adjusting Bricks':
        o = 0
    elif o == 'Crushing':
        o = 1
    elif o == 'Curing':
        o = 2
    elif o == 'Cutting bricks':
        o = 3
    elif o == 'Cutting Re-bar':
        o = 4  
    elif o == 'Drilling':
        o = 5
    elif o == 'Excavation':
        o = 6
    elif o == 'Fetching Mortar':
        o = 7
    elif o == 'Lashing Rope':
        o = 8
    elif o == 'Moving Stones':
        o = 9
    elif o == 'Placing Rebar':
        o = 10
    elif o == 'Re-bar':
        o = 11
    elif o == 'Welding':
        o = 12
    else:
        o = 13	
    p = st.selectbox("**Over_time**",('Yes','No'))
    if p == 'No':
        p = 0
    else:
        p = 1               
    submit = st.button('**Predict Performance Rating**')
    if submit: 
        prediction = emp_perf_model1.predict([[a,b,c,d,e,f,g,h,i,j,k,m,n,o,p]])
        prediction = int(prediction)
        if prediction == 0:
            st.warning("Worker's performance is AVERAGE😐")
        elif prediction == 1:
            st.success("Worker's performance is GOOD😍")
        else:
            st.error("Worker's performance is LOW😞")


if __name__ == '__main__':
    main()
