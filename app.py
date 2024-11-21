import streamlit as st
import pandas as pd
import pickle
import numpy as np

df=pickle.load(open('df.pkl','rb'))
stacking_regressor_model=pickle.load(open('stacking_regressor_model.pkl','rb'))


st.title('Laptop Price Predictor')
company=st.selectbox('Brand',df['Company'].unique())
type=st.selectbox('Type',df['TypeName'].unique())
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])
weight=st.number_input('Weight of laptop(in Kg)',min_value=0.1,max_value=5.0,format="%.2f")
touchscreen=st.selectbox('TouchScreen',['Yes','No'])
ips=st.selectbox('Ips Display',['Yes','No'])

screen_size=st.number_input('Screen Size(in inch)', min_value=10.0, max_value=20.0)

resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

cpu=st.selectbox('Cpu',df['CpuProcessor'].unique())
hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu=st.selectbox('GPU',df['GpuName'].unique())
os=st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    ppi=None

    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0

    if ips=='Yes':
        ips=1
    else:
        ips=0

    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])

    if screen_size >= 10.0:
        ppi = (X_res ** 2 + Y_res ** 2) ** 0.5 / screen_size
    else:
        st.error("Please enter a valid screen size greater than or equal to 10.")
        st.stop()

    if weight <= 0:
        st.error("Please enter a valid weight for the laptop. Weight should be greater than 0.")
        st.stop()

    query=np.array([company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os])
    # print(query)


    query_df = pd.DataFrame([query], columns=[
        'Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'IPS Panel',
        'PPI', 'CpuProcessor', 'HDD', 'SSD', 'GpuName', 'OS'
    ])

    # Make the prediction

    try:
        predicted_price = np.exp(stacking_regressor_model.predict(query_df))[0]
        st.title(f'The Predicted Price of this configuration is ₹{predicted_price:,.2f}')
    except Exception as e:
        st.error(f"Error in prediction: {e}")