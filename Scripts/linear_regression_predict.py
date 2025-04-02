import pandas as pd
import joblib
import matplotlib.pyplot as plt

def make_predictions(input_data, model_path, output_path, plot_path):
    data = pd.read_csv(input_data)
    model = joblib.load(model_path)

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
    X = data[features]

    predictions = model.predict(X)
    data['Predicted Final'] = predictions

    data.to_csv(output_path, index = False)
    print(f"Predictions saved to {output_path}")

    plt.figure(figsize = (10, 6))
    plt.plot(data['Final Exam'], label = "Actual Grades", color = 'blue', marker = 'o')
    plt.plot(data['Predicted Final'], label = "Predicted Grades", color = 'red', marker = 'x')
    plt.title("Actual vs Predicted Grades")
    plt.xlabel("Samples")
    plt.ylabel("Grades")
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    plt.show()
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    make_predictions("./Data/processed_data.csv", "./Models/linear_regression_model.pkl", "./Results/linear_regression_predictions.csv", "./Results/Plots/linear_regression_results.png")
