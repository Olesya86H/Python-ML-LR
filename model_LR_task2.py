#!/usr/bin/env python
# coding: utf-8

# In[1]:


#импортируем биб-ки, нужные для построения модели и графиков
import numpy as np #для работы с массивами и прочими мат.вычислениями
import pandas as pd #для работы с данными - датафреймы
import matplotlib.pyplot as plt #для работы с графиками
from sklearn.linear_model import LinearRegression #для построения моделей, линейная регрессия в данном случае, её задача - построить прогноз.


# In[2]:


data = pd.read_csv('data_for_regression.csv') #считываем исходный файл в набор данных data
data = data.dropna() #удаляем нулевые значения из считанного набора данных


# In[3]:


#график начальных данных - зависимость x от y:
x = data['x']
y = data['y']
plt.plot(x, y)
plt.show()


# In[4]:


from sklearn.model_selection import train_test_split #для разбиения исходной выборки на обучающую и тестовую выборки в соотношении 80%/20%.
num_features = ['x'] #просто для красоты, чтобы показать, что можно обращаться и так к столбцу
X_train, X_test, y_train, y_test = train_test_split(data[num_features], data['y'], test_size=0.2, shuffle=False)#80/20 80% - это тренировочные данные, 20% - тестовые


# In[5]:


df_x = pd.DataFrame(data[num_features]) #загоняем в датафрейм наш исходный столбец x


# In[6]:


df_x #просто смотрим на наш x исходный


# In[7]:


model_LR = LinearRegression().fit(X_train, y_train) # строим модель линейной регрессии и обучем её (обучение - команда .fit)


# In[8]:


i = df_x.tail(1) #берем последний x из файла с исходными данными
array_= np.array(i)
i = 13
while i <= 15:
    array_ = np.append(array_, i)
    i = i + 1
X_test = pd.DataFrame(array_)
prediction = model_LR.predict(X_test) #прогоняем нашу модель линейной регрессии по данным х от 15-й позиции до конца - получаем прогнозные значения y
df_test_y_pred = pd.DataFrame()#заводим новую оболочку под х и расчётный у (y назовём z)
df_test_y_pred["x"] = X_test #загоняем х в датафрейм
df_test_y_pred["y_pred"] = prediction #добавляем в наш НД расчетное значение y - загоняем его в датафрейм
#пишем в файл результат, колонки x и y_pred (y_pred - это целевая, или target-колонка, это прогнозный y):
df_test_y_pred.to_csv("data_regression_test_model_LR.csv", index=False)
print("Результаты прогноза модели, построенной методом линейной регрессии выгружены в файл data_regression_test_model_LR.csv")
print("Файл хранится в той же папке, откуда запускается эта IDE")


# In[9]:


#график прогнозируемых данных - зависимость x (от последнего значения x из файла с исходными данными до x = 15) от y (который посчитала модель линейной регрессии):
x = df_test_y_pred["x"]
y = df_test_y_pred["y_pred"]
plt.plot(x, y)
plt.show()

