import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, mean_squared_error, confusion_matrix



columns = ["Sample code number","Clump Thickness","Uniformity of Cell Size",
            "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", 
            "Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses", "Class"]
df = pd.read_csv(r"C:\Users\gerar\Downloads\breast-cancer-wisconsin.data", encoding='utf-8', names = columns)

df.drop(columns=['Sample code number','Bare Nuclei'], inplace=True)

df_x = df[["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape", 
            "Marginal Adhesion", "Single Epithelial Cell Size", 
            "Bland Chromatin","Normal Nucleoli","Mitoses"]] 

df_y = df["Class"]

Xtrain, Xtest, ytrain, ytest = train_test_split(df_x, df_y,random_state=1)

model = LogisticRegression(fit_intercept = True)
model.fit(Xtrain,ytrain)


pred_y = model.predict(Xtest)

acc = accuracy_score(ytest, pred_y)

#print("la varianza del modelo es: " + str(explained_variance_score(ytest, pred_y)))

print("accuracy of the model: "+ str(acc))


########################################33
print(cross_val_score(model,df_x,df_y, cv=10).mean())

ridgeModel = Ridge() # 
ridgeModel.fit(Xtrain, ytrain)
pred_yR = ridgeModel.predict(Xtest)
print(cross_val_score(ridgeModel, df_x, df_y, cv = 10).mean())
#print(accuracy_score(ytest, pred_yR))
ytest.values

