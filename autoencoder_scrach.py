import numpy as np
import random as ra
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

learning_rate= .000001

def calculate_loss(yhat,y):
    loss=0
    yhat=yhat[0]
    for i in range(len(y)):
        loss+=(yhat[i]-y[i])**2
    return loss/3.0

def sigmoid(x):	
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def activate(x):
    return sigmoid(x)

def output_layer(w,x):	
    return np.matmul(w,x)

def hidden_layer(w,x):	
    return np.matmul(w,x).reshape(1,3)

def gradient_outer(w,hidden, l):
    hidden = hidden[0]
    for i in range(len(w)):
        for j in range(len(w[i])):
            step=hidden[j]*2/3
            w[i][j]=float(w[i][j]-float((learning_rate*step)*l))
    return w

def gradient_inner(w,v,h, a, l):
    a = a[0]
    for i in range(len(w)):
        for j in range(len(w[i])):
            w[i][j]=w[i][j]-(2/3*l*(learning_rate*h[j]*sigmoid_derivative(a[j])))
    return w

def create_data():
    data = np.zeros((100,3))
    for i in range(100):
        temp = np.zeros((3,))
        for j in range(3):
            temp[j] = ra.gauss(2, .2)
        data[i] = temp
    return data
        

if __name__ =='__main__':
    #x=np.asarray([2,2,2]).reshape(1,3).astype(float)
    #w1=np.asarray([[.1,1, .3],[2,3.4,4.6],[7,1.4,.1]]).reshape(3,3).astype(float)
    #w2=np.asarray([[1.5,1.2,.7],[3,1.4,.12],[.14,.13,.1]]).reshape(3,3).astype(float)
    data = normalize(create_data())

    w1 = np.random.rand(3,3)
    w2 = np.random.rand(3,3)
    total_loss=100
    count = 0
    t_losses = []
    for i in range(5000):
        losses = []
        for j in range(100):
            rand = ra.randint(0, 90)
            x = data[rand]
            h = hidden_layer(x,w1)
            a = activate(h)
            yhat=output_layer(h,w2)
            new_loss=calculate_loss(yhat,x)
            total_loss = new_loss
            #if count%1000 == 0:
                #print("Loss "+str(total_loss))
            yhat=yhat[0]
            h=h[0]
            w2=gradient_outer(w2,a, total_loss)
            w1=gradient_inner(w1,x[0],h, a, total_loss)
            count +=1
            losses.append(total_loss)
        #if count == 500000:
            #break
        print(np.average(losses))
        t_losses.append(np.average(losses))
    print(count)
    print(total_loss)
    print(yhat)
    plt.plot(t_losses)
    plt.show()
                        
