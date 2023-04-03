# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: S.Sham Rathan
RegisterNumber: 212221230093
*/
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of citiy(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def computeCost(X,y,theta):
  m=len(y) 
  h=X.dot(theta)
  square_err=(h - y)**2
  return 1/(2*m) * np.sum(square_err)
 data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(X,y,theta)
def gradientDescent(X,y,theta,aplha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions-y))
    descent=aplha* 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))
  return theta,J_history
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(/Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions = np.dot(theta.transpose(),x)
  return predictions[0]
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))
predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```
## Output:
### Profit Prediction graph:
![image](https://user-images.githubusercontent.com/93587823/229563176-df427cbd-a789-49f7-aeaf-77185b881b5e.png)
### Compute Cost Value:
![image](https://user-images.githubusercontent.com/93587823/229276137-e4b93eba-1222-4d11-87c4-b43dc0ff6c1f.png)
### h(x) Value:
![image](https://user-images.githubusercontent.com/93587823/229275828-a2f05835-b2a4-4a25-80b6-1d9e57110d6f.png)
### Cost function using Gradient Descent Graph:
![image](https://user-images.githubusercontent.com/93587823/229563346-79c49ddf-c78a-42b5-aff1-14f01c42a9fe.png)
### Profit Prediction Graph:
![image](https://user-images.githubusercontent.com/93587823/229563455-2f68f96d-21f7-486f-a1ba-cfcd5c73dad1.png)
### Profit for the Population 35,000:
![image](https://user-images.githubusercontent.com/93587823/229276318-a1eb075b-704c-45ed-8f84-9b761c6cc32d.png)
### Profit for the Population 70,000:
![image](https://user-images.githubusercontent.com/93587823/229276377-f68a383d-459f-45e1-bc4b-5bbd4f1ae751.png)
## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
