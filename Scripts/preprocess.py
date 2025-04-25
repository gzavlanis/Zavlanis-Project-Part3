import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path, categorized_path, avg_path):
    data = pd.read_excel(input_path, header = [0, 1])
    data.columns = [' '.join(col).strip() for col in data.columns.values]

    data.replace('-', np.nan, inplace = True)
    data.replace(-1, np.nan, inplace = True)
    data.fillna(0, inplace = True)

    data['Final Exam'] = data[['Exams Final', 'Exams Repeat']].max(axis = 1)
    data.drop(columns = ['Exams Final', 'Exams Repeat'], inplace = True)
    data = data.loc[:, (data != 0).any(axis = 0)]

    bins = [-1, 5, 7, 10]
    labels = ["Low", "Medium", "High"]

    data.to_csv(output_path, index = False)
    print(f"Data saved to {output_path}")

    # Aggregate columns to work with average values for clustering
    avg_data = pd.DataFrame()
    avg_data["Homework Avg"] = data[[f"Homework Assignments {i}" for i in range(1, 5)]].mean(axis = 1)
    avg_data["Compulsory Activities Avg"] = data[[f"Compulsory Activities {i}" for i in range(1, 9)]].mean(axis = 1)
    avg_data["Optional Activities Avg"] = data[[f"Optional Activities {i}" for i in range(1, 11)]].mean(axis = 1)
    avg_data["Final Exam"] = data["Final Exam"]
    avg_data = avg_data.round(4)
    avg_data.to_csv(avg_path, index = False)
    print(f"AVG data saved to {avg_path}")

    data['Category'] = pd.cut(data['Final Exam'], bins = bins, labels = labels)
    data.to_csv(categorized_path, index = False)
    print(f"Categorized data saved to {categorized_path}")

if __name__ == "__main__":
    preprocess_data("../Data/grades.xlsx", "../Data/processed_data.csv", "../Data/categorized_data.csv", "../Data/avg_data.csv")
