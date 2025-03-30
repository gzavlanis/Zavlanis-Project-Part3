import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model(data_path, model_path):
    data = pd.read_csv(data_path)

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
        'Optional Activities 10',
    ]
    target = 'Final Exam'  # Use the combined column for grades
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # Split the dataset into training and testing sets

    model = DecisionTreeRegressor(max_depth = 5, random_state = 42) # Initialize Decision Tree Regressor
    model.fit(X_train, y_train) # Train the model
    y_pred = model.predict(X_test) # Make predictions on test data

    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Print performance report
    print(f"Model Performance:")
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Show actual vs. predicted values
    comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    print("\nActual vs Predicted (Top 10 Rows):")
    print(comparison.head(20))

    # Save the trained model
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    train_model("./Data/processed_data.csv", "./Models/decision_tree_model.pkl")
