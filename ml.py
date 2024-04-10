import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plt
import seaborn as sn
from  sklearn.linear_model import Perceptron 
from mlxtend.plotting import plot_decision_regions

student_data=pd.read_csv('collegePlace.csv')
# print(student_data.columns)
ndata=student_data[['Internships','CGPA', 'PlacedOrNot']]
# sn.scatterplot(x=ndata['Internships'],y=ndata['CGPA'],hue=ndata['PlacedOrNot'])
# plt.show()
print('before perceptron')
p=Perceptron()
X=ndata.iloc[:,0:2]
y=ndata.iloc[:,-1]
p.fit(X,y)
print(p.coef_)
plot_decision_regions(X.values, y.values, clf=p, legend=2)
print('after perceptron')
