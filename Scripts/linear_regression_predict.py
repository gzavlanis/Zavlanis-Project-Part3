import pandas as pd
import joblib

def make_predictions(input_data, model_path, output_path):
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

if __name__ == "__main__":
    make_predictions("./Data/processed_data.csv", "./Models/linear_regression_model.pkl", "./Results/linear_regression_predictions.csv")
