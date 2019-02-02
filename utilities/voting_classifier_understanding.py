import numpy as np 
from matplotlib import pyplot as plot 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y = True)

scaler = StandardScaler()
x = scaler.fit_transform(x)

classifier1 = DecisionTreeClassifier(criterion='gini', max_depth=4)
classifier2 = LogisticRegression( max_iter=200)
classifier3 = Perceptron(eta0 = 0.1, max_iter=200)

x_train,x_test, y_train,y_test = train_test_split(x,y,test_size = 0.3, stratify = y, random_state = 123)

classifier1.fit(x_train,y_train)
classifier2.fit(x_train,y_train)
classifier3.fit(x_train,y_train)

predict1 = classifier1.predict(x_test)
predict2 = classifier2.predict(x_test)
predict3 = classifier3.predict(x_test)

weight1 = 0.3
weight2 = 0.3
weight3 = 0.4
weights = [weight1,weight2,weight3]

predictions = np.array([predict1,predict1,predict3])

##why apply along axis..well bincount doesnt take into account any axis. it always works off flat arrays
## inour predictions, shape is 45, but when we have 3, we have shape 3,45. Where each classifier has
## their predictions along the rows of each columns.
## so [1,1],[2,1],[3,1] are the predictions for the 2nd element in x_test by classifer 1,2,3
## so [1,3],[2,3],[3,3] are the predictions for the 4th element in x_test by classifer 1,2,3
## in order for us to bincount , we use apply_along_axis and pass the weights and take the
## argmax

vote = np.apply_along_axis(lambda x:  np.argmax(np.bincount(x, weights = weights)), axis = 0, arr = predictions)

print(vote.shape, vote)
# np.apply_along_axis(lambda x : np.bincount(x, weights = weights),axis = 1, arr= predictions.T)
