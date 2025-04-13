Các thư viện đã được sử dụng:
- scikit-learn
- jupyterlab
- graphviz (cài thêm graphviz ở web: https://graphviz.org/download/)
- pandas
- numpy
- matplotlib
- seaborn
Data:
- ucimlrepo



Sử dụng các data:

-Install the ucimlrepo package:
pip install ucimlrepo

-Import the dataset into your code:
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
X = heart_disease.data.features 
y = heart_disease.data.targets 
  
# metadata 
print(heart_disease.metadata) 
  
# variable information 
print(heart_disease.variables)



-Install penguins data:
pip install palmerpenguins

-Import the dataset penguins:
from palmerpenguins import load_penguins
import pandas as pd

# Tải dữ liệu và lưu vào DataFrame
df = load_penguins()

# In ra 5 dòng đầu tiên để kiểm tra
print(df.head())



Install the ucimlrepo package 
pip install ucimlrepo
Import the dataset into your code 
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
chronic_kidney_disease = fetch_ucirepo(id=336) 
  
# data (as pandas dataframes) 
X = chronic_kidney_disease.data.features 
y = chronic_kidney_disease.data.targets 
  
# metadata 
print(chronic_kidney_disease.metadata) 
  
# variable information 
print(chronic_kidney_disease.variables) 