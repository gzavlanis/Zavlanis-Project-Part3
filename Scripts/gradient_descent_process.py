from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import joblib

def train_model(input_data, model_path, plot_path):
    avg_data = pd.read_csv(input_data)
    X = avg_data[['Homework Avg', 'Compulsory Activities Avg', 'Optional Activities Avg']]
    y = avg_data['Final Exam']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    print("Scaled inputs: ", X_scaled)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    model = SGDRegressor(max_iter = 1000, tol = 1e-3, learning_rate = 'optimal')  # Create a SGDRegressor object
    model.fit(X_train, y_train)  # Fit the model to the data
    predictions = model.predict(X_test)  # Make predictions
    train_predictions = model.predict(X_train)
    mse = mean_squared_error(y_test, predictions)  # Calculate the mean squared error

    print("Predicted Final Grades:", predictions[:10])
    print("\nOptimized Parameters (Theta):", model.coef_)
    print("\nBias Term (Intercept):", model.intercept_)
    print("\nMean Squared Error:", mse)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Plot the predicted tests
    plt.figure(figsize = (8, 6))
    plt.scatter(y_test, predictions, color = "blue", alpha = 0.5, label = "Predictions")
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color = "red", linestyle = "--", label = "Perfect Fit")
    plt.xlabel("Actual Final Grades")
    plt.ylabel("Predicted Final Grades")
    plt.title("Predicted vs Actual Final Grades")
    plt.legend()
    plt.savefig(plot_path)
    plt.show()

    # Plot train data predictions
    plt.figure(figsize = (8, 6))
    plt.scatter(y_train, train_predictions, color = "blue", alpha = 0.5, label = "Predictions")
    plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color = "red", linestyle = "--", label = "Perfect Fit")
    plt.xlabel("Actual Final Grades (train part)")
    plt.ylabel("Predicted Final Grades")
    plt.title("Predicted vs Actual Final Grades (Train part)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    train_model("../Data/avg_data.csv", "../Models/linear_regression_model.pkl", '../Results/Plots/Actual_vs_Predicted_Exam_Marks.png')
