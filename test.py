import streamlit as st
import pandas as pd
import yfinance as yf 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


app_name= "Stock Market Prediction app"
st.title(app_name)
st.subheader("This app is created to predict the stock market price of the selected company")

st.image("data:image/jpeg;base64,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",width=1000)

start_date = st.sidebar.date_input('start date',date(2020,1,1))
end_date = st.sidebar.date_input('End date',date(2020,12,31))

ticker_list = ["AAPL","MSFT","GOOG","GOOGL","meta","TESLA","NVDA","ADBE","PYPL","INTC","CMCSA","NFLX","PEP"]
ticker = st.sidebar.selectbox('select the company',ticker_list)



data = yf.download(ticker, start=start_date,end=end_date,auto_adjust=False, )
data.reset_index(inplace=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns =['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]
    
    st.write(data)
    
    st.header('Data Visualization')
    st.subheader('Plot of the data')
    
    fig=px.line(data, x='Date_', y=data.columns,title='Closing price of the stock',width=1000,height=600)
    st.plotly_chart(fig)
    
    selected_column=st.selectbox('select the column to be used for forecasting',data.columns[1:])
    
    data= data[['Date_',selected_column]]

    st.write(data)
    
    
    st.header('Is data Stationary?')
    st.write('**Note:** If p-value is less than0.05,then data is stationary')
    st.write(adfuller(data[selected_column])[1]<0.05)
    
    st.header('Decomposition of the data')
    decomposition=seasonal_decompose(data[selected_column],model='additive',period=12)
    st.write(decomposition.plot())
    
    
    st.write("## Ploting the decomposition in plotly")
    st.plotly_chart(px.line(x=data["Date_"], y=decomposition.trend,title='Trend',width=1200,height=400, labels={'x':'Date','y':'price'}))
    st.plotly_chart(px.line(x=data["Date_"], y=decomposition.seasonal,title='Seasonal',width=1200,height=400, labels={'x':'Date','y':'price'}))
    st.plotly_chart(px.line(x=data["Date_"], y=decomposition.resid,title='Residuals',width=1200,height=400, labels={'x':'Date','y':'price'}))
    
    p =st.slider('select the value of p',0,5,2)
    d =st.slider('select the value of d',0,5,1)
    q =st.slider('select the value of q',0,5,2)
    seasonal_order= st.number_input('select the value of seasonal p',0,24,12)
    
    
    model = sm.tsa.statespace.SARIMAX(data[selected_column],order=(p,d,q),seasonal_order=(p,q,d,seasonal_order))
    model=model.fit(disp=-1)
    
    st.header('Model Summary')
    st.write(model.summary())
    st.write("---")
    st.write("<p style='color:green; font-size: 50px; font-weight: bold ; '> Forecasting the data</p>",unsafe_allow_html=True)
    
    forecast_period= st.number_input('Selected the number of days to forecast',1,365,10)
    
    predictions= model.get_prediction(start=len(data)-len(data),end=len(data)+forecast_period-1)
    predictions=predictions.predicted_mean
    
   # st.write("Data columns:",data.columns)
    
    start_date= data['Date_'].iloc[-1]+pd.Timedelta(days=1)
    end_date = start_date + pd.Timedelta(days=forecast_period-1)
    st.write(predictions)
    
    predictions.index = pd.date_range(start=end_date,periods=len(predictions),freq='D')
    predictions =pd.DataFrame(predictions)
    predictions.insert (0,'Date_',predictions.index)
    predictions.reset_index(drop=True,inplace=True)
    st.write("## predictions", predictions)
    st.write("## Actual Data",data) 
    st.write("---")
    #st.write("Columns:", list(predictions.columns))
 
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=data["Date_"],y=data[selected_column],mode='lines',name='Actual',line=dict(color='blue')))
   
    fig.add_trace(go.Scatter(x=predictions["Date_"],y=predictions['predicted_mean'],mode='lines',name='predicted',line=dict(color='red')))
    
    fig.update_layout(title='Actual vs Predicted',xaxis_title='Date_', yaxis_title='price',width=1200,height=400)
    
    st.plotly_chart(fig)
    
    
    
    
#     show_plots = Flase
#     if st.button('show separate plts'):
#         if not show_plots:
#             st.write(
#     px.line(
#         data,
#         x=data["Date_"],
#         y=selected_column,
#         title="Actual",
#         width=1200,
#         height=400,
#         labels={'x': 'Date_', 'y': 'price'}
#     )
# )
#             st.write(
#     px.line(
#         data,
#         x=predictions["Date_"],
#         y=predictions['predicted_mean'],
#         title="predicted",
#         width=1200,
#         height=400,
#         labels={'x': 'Date_', 'y': 'price'}
#     )
# )
# show_plots= True 
# else:
#     show_plots= False
show_plots = False  # Fixed typo and removed indent

if st.button('show separate plots'):
    x_vals = data["Date_"]
    y_vals = data[selected_column]

    min_len = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]

    st.write(px.line(
        x=x_vals,
        y=y_vals,
        title="Actual",
        width=1200,
        height=400,
        labels={'x': 'Date_', 'y': 'price'}
    ))

    # Actual plot
    x_vals = predictions["Date_"]
    y_vals = predictions["predicted_mean"]

    min_len = min(len(x_vals), len(y_vals))
    x_vals = x_vals[:min_len]
    y_vals = y_vals[:min_len]

    st.write(px.line(
        x=x_vals,
        y=y_vals,
        title="Predicted",
        width=1200,
        height=400
    ))
 
show_plots = True

# Inject CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #e6f7ff;
    }
    /* Optional: style full page body */
    body {
        background-color: #e6f7ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)



