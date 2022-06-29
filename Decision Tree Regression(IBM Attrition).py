#decision tree REGRESSION

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
df=pd.read_csv("C:\\Users\\HEMANTH\\OneDrive\\Desktop\\salary_decision.csv")
x=df[["Experience"]]
y=df[["salary"]]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0)
from sklearn.tree import DecisionTreeRegressor
D_reg=DecisionTreeRegressor()
D_reg.fit(x_train,y_train)
print(D_reg.score(x_test,y_test))
from sklearn import tree
print(x_train,y_train)
print(D_reg.predict([[11]]))
print(D_reg.predict([[12]]))
print(D_reg.predict([[13]]))
print(D_reg.predict([[14]]))
print(D_reg.predict([[20]]))
_=tree.plot_tree(D_reg,feature_names="Experience",filled=True,rounded=True)
plt.show()
print(D_reg.get_params())