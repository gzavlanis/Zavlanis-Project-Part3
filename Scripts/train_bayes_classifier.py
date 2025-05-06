import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def train_bayes_classifier(data_file, model_file):
    data = pd.read_csv(data_file)
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
    target = 'Category'  # Use the combined column for grades
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42) # Split the dataset into training and testing sets

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    report = classification_report(y_test, y_pred, output_dict = True)
    report_df = pd.DataFrame(report).transpose()

    print(report_df)
    plt.figure(figsize = (8, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot = True, cmap = "Blues")
    plt.title("Classification Report Heatmap")
    plt.show()
    plt.savefig("../Results/Plots/classification_report_heatmap.png")

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_bayes_classifier("../Data/categorized_data.csv", "../Models/naive_bayes_model.pkl")
