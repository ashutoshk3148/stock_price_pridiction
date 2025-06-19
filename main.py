import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st

model = load_model('C:\Hp\Desktop\stock\Stock Prediction Model.keras')

st.header('Stock Market Predictor')
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

data = yf.download(stock, satrt, end)

st.subheader('Stock Data')
st.write(data)

data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_100_days = data_tarin.tail(100)
data_test = pd.concat([pas_100_days, data_train], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

x = []
y = []

for i in range(100,data_train_scale.shape[0]):
    x.append(data_train_scale[i-100:i])
    y.append(data_train_scale[i,0])

x,y = np.array(x), np.array(y)