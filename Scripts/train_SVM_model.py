import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def train_model(input_data, model_path, cm_path, roc_path, feature_imp_path):
    data = pd.read_csv(input_data)
    data['Dropout Risk'] = np.where(data['Final Exam'] < 5, 1, 0)
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
    y = data['Dropout Risk']

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42)

    model = SVC(kernel = "linear", C = 1, gamma = 0.1)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))
    with open("../Results/Reports/dropout_prediction_report.txt", "w") as file:
        file.write(classification_report(y_test, predictions))

    with open(model_path, "wb") as file:
        pickle.dump(model, file)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize = (6, 5))
    sns.heatmap(cm, annot = True, fmt = "d", cmap = "Blues", xticklabels = ["Not At Risk", "At Risk"], yticklabels=["Not At Risk", "At Risk"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Dropout Prediction")
    plt.show()
    plt.savefig(cm_path)

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure(figsize = (6, 5))
    plt.plot(fpr, tpr, color = "darkorange", label = f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color = "gray", linestyle = "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Dropout Prediction")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig(roc_path)

    feature_importance = np.abs(model.coef_[0])
    # Plot Feature Importance
    plt.figure(figsize = (6, 5))
    plt.barh(features, feature_importance, color = "green")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Feature Importance in Dropout Prediction")
    plt.show()
    plt.savefig(feature_imp_path)

if __name__ == "__main__":
    train_model("../Data/processed_data.csv", "../Models/svm_dropout_model.pkl", "../Results/Plots/SVM_confusion_matrix.png", "../Results/Plots/SVM_ROC_Curve.png", "../Results/Plots/SVM_feature_importance_chart_png")
