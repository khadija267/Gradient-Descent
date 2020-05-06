import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#to make same numbers appear each time
np.random.seed(7)
#calculate gradient descent for mean squared error(performane mertic)
#X=> predictor features array
#y=> outcomes in the data
#W=>weights
#b=>intercept
#alpha=>learning rate
def MSE(X,y,W,b,alpha=0.005):
    #y_hat is predicted value=> this is Regression eqn
    y_hat=np.matmul(X,W)+b
    Error=y-y_hat
    #updating to obtain new values of W and b
    W_new=W+alpha*np.matmul(Error,X)
    b_new=b+alpha*Error.sum()
    return W_new,b_new
################33TEST#########################
def miniBatch(X, y, batch_size = 20, alpha = 0.005, iter_nums = 25):
    #X.shape[0]=>no of rows
    points_num=X.shape[0]
    #initialize weights as no of colums os X with same no of rows
    W=np.zeros(X.shape[1])
    #initialize intercept =0
    b=0
    #hstack=>Stack arrays in sequence horizontally (column wise)=> 1 &1=> 11
    #hstack outputs an array & get 1 argument
    reg_coeff=[np.hstack((W,b))]
    #iterate
    for e in range(points_num):
        batch=np.random.choice(range(points_num),batch_size)
        x_batch=X[batch,:]
        y_batch=y[batch]
        W,b=MSE(x_batch,y_batch,W,b,alpha)
        reg_coeff.append(np.hstack((W,b)))
    return reg_coeff
########run###########3
if __name__=="__main__":
    data=pd.read_csv('data.csv')
    #X is all row all col except last col
    X=data.iloc[:,:-1].values
    #y is all row and last col
    y=data.iloc[:,-1].values
    reg_coeff=miniBatch(X,y)
    #plot
    X_min=X.min()
    X_max=X.max()
    i=len(reg_coeff)
    for W,b in reg_coeff:
        i-=1
        plt.plot([X_min, X_max],[X_min * W + b, X_max * W + b], color = 'red')
    plt.scatter(X, y, zorder = 3)    
    
    
    
    
    
    
