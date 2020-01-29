from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

#считываем файл
data = pd.read_csv('transport_data.csv')
df1 = data[(data.label == '0') | (data.label == '1') | (data.label == '2')]

df1.label = df1.label.astype(int)

model = RandomForestClassifier()
model.fit(df1.iloc[:, :2], df1.iloc[:, 4])

y_pred = model.predict(data[(data.label == '?')].iloc[:, :2])

with open('your_file.txt', 'w') as f:
    for item in y_pred:
        f.write("%s\n" % int(item))