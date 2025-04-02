import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_data(input_path, output_path, categorized_path):
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

    data['Category'] = pd.cut(data['Final Exam'], bins = bins, labels = labels)
    data.to_csv(categorized_path, index = False)
    print(f"Categorized data saved to {categorized_path}")

if __name__ == "__main__":
    preprocess_data("./Data/grades.xlsx", "./Data/processed_data.csv", "./Data/categorized_data.csv")
