
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
import joblib

def train_lstm_model(train_csv_path, model_save_path="saved_model.h5", scaler_save_path="saved_scaler.pkl"):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
    from tensorflow.keras.callbacks import EarlyStopping

    LOOK_BACK = 60
    EPOCHS = 100
    BATCH_SIZE = 32

    train_df = pd.read_csv(train_csv_path)
    train_close = train_df['close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(train_close)

    x_train, y_train = [], []
    for i in range(LOOK_BACK, len(scaled_data)):
        x_train.append(scaled_data[i - LOOK_BACK:i])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    model = Sequential([
        Input(shape=(x_train.shape[1], 1)),
        LSTM(100, return_sequences=True),
        Dropout(0.3),
        LSTM(100, return_sequences=False),
        Dropout(0.3),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

    early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stop], verbose=0)

    model.save(model_save_path)
    joblib.dump(scaler, scaler_save_path)

    return model, history

def predict_lstm_model(test_csv_path, model_path="saved_model.h5", scaler_path="saved_scaler.pkl", look_back=60):
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)

    test_df = pd.read_csv(test_csv_path)
    test_close = test_df['close'].values.reshape(-1, 1)

    # Add last look_back days from test itself for simplicity
    combined_data = np.concatenate((test_close[:look_back], test_close), axis=0)
    input_data_scaled = scaler.transform(combined_data)

    x_test, y_test = [], []
    for i in range(look_back, len(input_data_scaled)):
        x_test.append(input_data_scaled[i - look_back:i])
        y_test.append(input_data_scaled[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_pred_scaled = model.predict(x_test)

    y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_pred_original = scaler.inverse_transform(y_pred_scaled)

    mae = mean_absolute_error(y_test, y_pred_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_scaled))
    r2 = r2_score(y_test_original, y_pred_original)
    mape = np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100

    return {
        "mae_scaled": mae,
        "rmse_scaled": rmse,
        "r2_original": r2,
        "mape_original": mape,
        "y_test_original": y_test_original,
        "y_pred_original": y_pred_original
    }
