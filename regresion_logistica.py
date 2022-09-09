import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.preprocessing import MaxAbsScaler
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import explained_variance_score, mean_squared_error, confusion_matrix

def plot_learning_curve(               # función para imprimir el accuracy del modelo en una gráfica
    estimator,
    title,
    X,
    y,
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    scoring=None,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    plt.grid()
    plt.plot(
        train_sizes, train_scores_mean, "o-", color="r", label="Training score"
    )
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")



columns = ["Sample code number","Clump Thickness","Uniformity of Cell Size",
            "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", 
            "Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses", "Class"]
df = pd.read_csv(r"C:\Users\gerar\Downloads\breast-cancer-wisconsin.data", encoding='utf-8', names = columns)  #1,7, ult

df.drop(columns=['Sample code number','Bare Nuclei'], inplace=True)

df_x = df[["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape", 
            "Marginal Adhesion", "Single Epithelial Cell Size", 
            "Bland Chromatin","Normal Nucleoli","Mitoses"]] 

df_y = df["Class"]

 ########################### separación de los datos y entrenamiento del modelo ###############################
Xtrain, Xtest, ytrain, ytest = train_test_split(df_x.values, df_y.values,random_state=1)
model = LogisticRegression(fit_intercept = True)
model.fit(Xtrain,ytrain)
pred_y = model.predict(Xtest)
acc = accuracy_score(ytest, pred_y)

########################## accuracy ########################
print("accuracy of the model: "+ str(acc))
print("score of the cross validation: " + str(cross_val_score(model,df_x,df_y, cv=10).mean()))
print("Matrix of confusion: \n" + str(confusion_matrix(ytest, pred_y)))

################### regularización #######################
ridgeModel = Ridge() # 
ridgeModel.fit(Xtrain, ytrain)
pred_yR = ridgeModel.predict(Xtest)
print("score of the cross validation for the Ridge model: " + str(cross_val_score(ridgeModel, df_x, df_y, cv = 10).mean()))

#################### predicción individual #############
pred = model.predict([[5,1,1,1,2,3,1,1]])
print("prediction for input [5,1,1,1,2,3,1,1]: " + str(pred))

######################## graficar el accuracy #################
fig, axes = plt.subplots(1, 1, figsize=(15, 7))
title = r"Score"
estimator = model
plot_learning_curve(
    estimator, title, df_x, df_y, ylim=(0.7, 1.01), cv=15, n_jobs=4 
)
plt.show()