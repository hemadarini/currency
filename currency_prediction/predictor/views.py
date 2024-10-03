from django.shortcuts import render
from django.http import JsonResponse
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load the model and scaler
model = load_model('C:/Users/HP LAPTOP/Desktop/currency/lstm_currency_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        data = request.POST.get('data')
        print("Received data:", data)

        # Convert input
        price_data = np.array(data.split(',')).astype(float).reshape(-1, 1)
        if len(price_data) <29:
            return JsonResponse({'error': 'Not enough data. Provide at least 30 prices.'})

        scaled_data = scaler.fit_transform(price_data)
        time_step = 29
        X = []
        for i in range(len(scaled_data) - time_step):
            X.append(scaled_data[i:i + time_step])

        X = np.array(X)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        predictions = model.predict(X)
        predictions = scaler.inverse_transform(predictions)

        print("Predictions:", predictions)  # Check if predictions are generated
        return JsonResponse({'predictions': predictions.flatten().tolist()})