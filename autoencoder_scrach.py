import numpy as np
import random as ra
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

learning_rate= .01
input_size = 3
hidden_size = 2

def sigmoid(x):
    return 1/(1+np.exp(x))

def compute_loss(a, b):
    loss = 0
    for i in range(len(a)):
        loss += (a[0][i]-b[0][i])**2
    return loss/len(a)

def backpropogate(w1, w2, vector, out,hidden, b1, b2):
    delta = np.zeros((input_size,1))
    adj_bias2 = 0.0
    for i in range(input_size):
        delta[i][0] = (out[i][0] - vector[0][i])*out[i][0]*(1-out[i][0])
        adj_bias2 -= learning_rate*delta[i][0]*b2
    adj_matrix2 = np.matmul(delta, hidden.transpose())
    adj_matrix2 = adj_matrix2 +(-1*learning_rate)
    hidden2 = (hidden*-1)+1
    delta2 = np.matmul(delta.transpose(),w2)
    adj_bias1 = 0
    for i in range(hidden_size):
        delta2[0][i] = delta2[0][i]*hidden2[i][0]*hidden[i][0]
        adj_bias1 -= learning_rate *delta2[0][i]*b1
    adj_matrix = np.matmul(delta2.transpose(), vector)
    adj_matrix = adj_matrix*(-1*learning_rate)
    w1 = w1 + adj_matrix
    w2 = w2 + adj_matrix2
    b1 += adj_bias1
    b2 += adj_bias2
    return w1, w2, b1, b2

def feedforward(w1, w2, b1, b2, vector):
    hidden = np.matmul(w1, vector.transpose())
    hidden = hidden+b1
    hidden = sigmoid(hidden)
    out = np.matmul(w2, hidden)
    out = out + b2
    out = sigmoid(out)
    return out, hidden

def generate_normal():
    a = np.zeros((100, 3))
    for i in range(100):
        a[i][0] = ra.gauss(2, .01)
        a[i][1] = ra.gauss(13, .01)
        a[i][2] = ra.gauss(25, .1)
    return a

if __name__ == '__main__':
    a = generate_normal()
    scalar = MinMaxScaler()
    scalar.fit(a)
    a = scalar.transform(a)
    w1 = np.random.rand(2,3)
    w2 = np.random.rand(3,2)
    b1 = .015
    b2 = .015
    total_losses = np.zeros((1000,))
    for j in range(1000):
        losses = np.zeros((100,))
        for i in range(100):
            rand = ra.randint(0, len(a)-1)
            rand_instance = np.asarray([a[rand]])
            b, h = feedforward(w1, w2, b1, b2, rand_instance)
            losses[i] = compute_loss(rand_instance, b)
            w1, w2, b1, b2 = backpropogate(w1, w2, rand_instance, b, h, b1,b2)
        total_losses[j] = np.average(losses)
    plt.plot(total_losses)
    plt.show()
