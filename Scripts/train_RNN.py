import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def train_RNN(input_data, model_path, loss_plot_path, results_plot_path):
    data = pd.read_csv(input_data)
    features = [
        'Homework Assignments 1',
        'Homework Assignments 2',
        'Homework Assignments 3',
        'Homework Assignments 4',
        'Compulsory Activities 1',
        'Compulsory Activities 2',
        'Compulsory Activities 3',
        'Compulsory Activities 4',
        'Compulsory Activities 5',
        'Compulsory Activities 6',
        'Compulsory Activities 7',
        'Compulsory Activities 8',
        'Optional Activities 1',
        'Optional Activities 2',
        'Optional Activities 3',
        'Optional Activities 4',
        'Optional Activities 5',
        'Optional Activities 6',
        'Optional Activities 7',
        'Optional Activities 8',
        'Optional Activities 9',
        'Optional Activities 10'
    ]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[features])

    X, y = [], []
    sequence_length = len(features)

    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:i + sequence_length])
        y.append(data_scaled[i + sequence_length][-1])

    X, y = np.array(X), np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    model = Sequential([
        LSTM(64, activation = 'relu', return_sequences = True, input_shape = (sequence_length, len(features))),
        LSTM(32, activation = 'relu'),
        Dense(1, activation = 'linear')
    ])
    model.compile(optimizer = 'adam', loss = 'mse', metrics = ['mae'])
    history = model.fit(X_train, y_train, epochs = 20, batch_size = 8, validation_data = (X_test, y_test), verbose = 1)
    model.save(model_path)

    # Plot loss
    plt.plot(history.history["loss"], label = "Training Loss")
    plt.plot(history.history["val_loss"], label = "Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Process - Loss Over Epochs")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(loss_plot_path)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Model MAE: {mae:.4f}')

    comparison_df = pd.DataFrame({ "Actual": y_test.flatten(), "Predicted": predictions.flatten() })
    print(comparison_df.head(10))

    plt.scatter(range(len(y_test)), y_test, label = "Actual Grades", color = "blue", alpha = 0.7)
    plt.scatter(range(len(predictions)), predictions, label = "Predicted Grades", color = "red", alpha = 0.7)
    plt.xlabel("Student Index")
    plt.ylabel("Grade")
    plt.title("Actual vs Predicted Grades (Scaled)")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(results_plot_path)

if __name__ == "__main__":
    train_RNN('../Data/processed_data.csv', '../Models/RNN_model.keras', '../Results/Plots/RNN_loss.png', '../Results/Plots/RNN_predictions.png')
