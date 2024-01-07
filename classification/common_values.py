from pandas import read_csv, DataFrame
from numpy import ndarray
from matplotlib.pyplot import figure, show, savefig
from sklearn.model_selection import train_test_split
from pandas import read_csv

data: DataFrame = read_csv('classification/data/class_pos_covid/data_preparation/feature_selection/lowvar_redudant.csv')

# Filter the data where "Credit_Score" equals 1
filtered_data = data[data['CovidPos'] == 0]

# Get the median value of every single variable
median_values = filtered_data.median()

print(median_values)