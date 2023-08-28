import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv("housing.csv")

# veri seti hakkında bilgiler ediniyoruz.
# print(df.info())

# NaN verileri kaldırıyoruz
df.dropna(inplace=True)

# Verilerimizi bağımlı ve bağımsız değişkenlerimize ayırdık. Target değerimizi belirledik
inputs = df.drop(['median_house_value'], axis=1)
target = df['median_house_value']

# train ve test verilerimizi de ayırdık
x_train,x_test,y_train,y_test = train_test_split(inputs,target,test_size=0.2)

# eğitim verilerimi toplu bir hale getiriyorum
train_data = x_train.join(y_train)

#print(train_data)

