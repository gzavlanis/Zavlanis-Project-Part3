import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_model(input_data, model_path, confusion_matrix_path, roc_curve_path):
    avg_data = pd.read_csv(input_data)
    avg_data['Performance Label'] = np.where(avg_data['Final Exam'] >= 5, 1, 0)
    X = avg_data[['Homework Avg', 'Compulsory Activities Avg', 'Optional Activities Avg', 'Final Exam']]
    y = avg_data['Performance Label']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print("Model Accuracy:", accuracy)

    # Confusion matrix visualization
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot = True, cmap = "Blues", fmt = "d")
    plt.xlabel("Predicted Labels")
    plt.ylabel("Actual Labels")
    plt.title("Confusion Matrix")
    plt.savefig(confusion_matrix_path, dpi = 300)
    plt.show()

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Detailed classification report
    report = classification_report(y_test, predictions)
    print("Classification Report:\n", report)
    with open("../Results/Reports/classification_report.txt", "w") as file:
        file.write(report)
    print("Classification report saved as 'classification_report.txt'")

    # Compute ROC Curve
    y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for class 1
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize = (8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color = "red", linestyle = "--")  # Baseline
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(roc_curve_path, dpi = 300)
    plt.show()

if __name__ == "__main__":
    train_model("../Data/avg_data.csv", "../Models/linear_regression_model.pkl", "../Results/Plots/Logistic_regression_conf_matrix.png", "../Results/Plots/ROC_Curve.png")
