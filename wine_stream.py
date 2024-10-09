import pickle
import streamlit as st
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


wine_model = pickle.load(open('wine.sav', 'rb'))

st.title('PREDIKSI KUALITAS WINE')


test_inputs = np.array([
    [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4],
    [6.3, 0.3, 0.34, 1.6, 0.049, 16.0, 48.0, 0.998, 3.47, 0.5, 9.5]
])
test_actuals = np.array([5, 6])  


fixed_acidity = st.text_input('Fixed Acidity')
volatile_acidity = st.text_input('Volatile Acidity')
citric_acid = st.text_input('Citric Acid')
residual_sugar = st.text_input('Residual Sugar')
chlorides = st.text_input('Chlorides')
free_sulfur_dioxide = st.text_input('Free Sulfur Dioxide')
total_sulfur_dioxide = st.text_input('Total Sulfur Dioxide')
density = st.text_input('Density')
pH = st.text_input('pH')
sulphates = st.text_input('Sulphates')
alcohol = st.text_input('Alcohol')


quality = ''
mae, mse, rmse = '', '', ''


if st.button('Predict Quality'):
    
    try:
        input_data = np.array([[float(fixed_acidity), float(volatile_acidity), float(citric_acid),
                                float(residual_sugar), float(chlorides), float(free_sulfur_dioxide),
                                float(total_sulfur_dioxide), float(density), float(pH),
                                float(sulphates), float(alcohol)]])
        
        
        prediction = wine_model.predict(input_data)
        
    
        quality = f'Prediksi Kualitas Wine adalah: {prediction[0]}'
        
        
        test_predictions = wine_model.predict(test_inputs)
        
        
        mae_value = mean_absolute_error(test_actuals, test_predictions)
        mae = f'MAE (Mean Absolute Error): {mae_value:.2f}'
        
        
        mse_value = mean_squared_error(test_actuals, test_predictions)
        mse = f'MSE (Mean Squared Error): {mse_value:.2f}'
        
        
        rmse_value = np.sqrt(mse_value)
        rmse = f'RMSE (Root Mean Squared Error): {rmse_value:.2f}'
        
    except ValueError:
        quality = "Please enter valid numerical values."


st.success(quality)


if mae and mse and rmse:
    st.write(mae)
    st.write(mse)
    st.write(rmse)
