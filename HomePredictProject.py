import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression


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

# ocean_proximity sütunundaki kategorik verileri, One-Hot Encoding yöntemiyle dönüştürür.
# One-Hot Encoding, kategorik verileri sayısal verilere dönüştürmek için kullanılır.
# Her farklı kategori bir sütun oluşturur ve bu sütunlar yalnızca 0 veya 1 içerebilir.
train_data=train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

# eğitim verilerindeki sütunların histogramlarını oluşturduk

#train_data.hist()
#plt.show()

# Bir ısı haritası, bir matrisin hücrelerini renklerle doldurarak veri korelasyonunu daha açık bir şekilde görmenizi sağlar.
# Korelasyon, iki değişken arasındaki ilişkinin gücünü ve yönünü ölçer.
# Pozitif bir korelasyon (yaklaşık 1) iki değişken arasında doğru orantılı bir ilişkiyi gösterir. Bir değişken artarken diğeri de artar.
# Negatif bir korelasyon (yaklaşık -1) iki değişken arasında ters orantılı bir ilişkiyi gösterir. Bir değişken artarken diğeri azalır.
# Korelasyon katsayısı yaklaşık olarak 0 ise, iki değişken arasında güçlü bir ilişki yok demektir.

# plt.figure(figsize=(15,8))
# sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")
# plt.show()

# Belirli sütunlardaki verilerin logaritma dönüşümünü yapmaktadır.
# Logaritma dönüşümü, verilerin dağılımını değiştirerek ve bazı durumlarda
# verilerin daha normal dağılımlı hale getirilmesine yardımcı olan bir veri ön işleme yöntemidir.

# Bu tür bir dönüşüm, büyük değerlere sahip verileri ölçeklendirir ve
# ayrıca sıfır veya negatif değerlerle başa çıkmak için +1 ekler.
train_data['total_rooms']= np.log(train_data['total_rooms']+1)
train_data['total_bedrooms']= np.log(train_data['total_bedrooms']+1)
train_data['population']= np.log(train_data['population']+1)
train_data['households']= np.log(train_data['households']+1)

# Saçılım grafiği, iki değişkenin (latitude ve longitude) ilişkisini ve aynı zamanda üçüncü bir değişkenin (median_house_value)
# renk skalası ile gösterir.
# plt.figure(figsize=(15,8))
# sns.scatterplot(x='latitude', y='longitude', data=train_data, hue='median_house_value', palette='coolwarm')
# plt.show()

train_data['bedroom_ratio']=train_data['total_bedrooms']/ train_data['total_rooms']
train_data['households_rooms']= train_data['total_rooms']/ train_data['households']

x_train,y_train,=train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
model = LinearRegression()

model.fit(x_train,y_train)

test_data = x_test.join(y_test)

test_data['total_rooms']= np.log(test_data['total_rooms']+1)
test_data['total_bedrooms']= np.log(test_data['total_bedrooms']+1)
test_data['population']= np.log(test_data['population']+1)
test_data['households']= np.log(test_data['households']+1)

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis=1)

test_data['bedroom_ratio']=test_data['total_bedrooms']/ test_data['total_rooms']
test_data['households_rooms']= test_data['total_rooms']/ test_data['households']



x_test,y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value']


print(model.score(x_test,y_test))