import pandas as pd
import numpy as np

def preprocess_data(input_path, output_path):
    data = pd.read_excel(input_path, header = [0, 1])
    data.columns = [' '.join(col).strip() for col in data.columns.values]

    data.replace('-', np.nan, inplace = True)
    data.replace(-1, np.nan, inplace = True)
    data.fillna(0, inplace = True)

    data['Final Exam'] = data[['Exams Final', 'Exams Repeat']].max(axis = 1)
    data.drop(columns = ['Exams Final', 'Exams Repeat'], inplace = True)
    data = data.loc[:, (data != 0).any(axis = 0)]

    data.to_csv(output_path, index = False)
    print(f"Data saved to {output_path}.")

if __name__ == "__main__":
    preprocess_data("./Data/grades.xlsx", "./Data/processed_data.csv")
