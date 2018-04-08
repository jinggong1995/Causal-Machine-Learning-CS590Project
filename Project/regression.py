import numpy as np
from sklearn.linear_model import Ridge
from numpy import linalg as LA



def gen_data(alpha, beta1,beta2,gama1,gama2,gama3,n):
    uz = np.random.standard_normal(n)
    ux = np.random.standard_normal(n)
    uy = np.random.standard_normal(n)
    uw = np.random.standard_normal(n)
    z = uz
    x = alpha*z + ux
    w = beta1 * x + beta2 * z + uw
    y = gama1 * x + gama2 * z + gama3 * w + uy
    data = np.zeros((n, 4))
    data[:, 0] = x
    data[:, 1] = z
    data[:, 2] = w
    data[:, 3] = y
    return data


def regress_y(data):
    x = data[:, :3]
    y = data[:, 3]
    m = np.linalg.inv(np.dot(np.transpose(x),x))
    n = np.dot(m, np.transpose(x))
    beta_y = np.dot(n,y)
    return beta_y

def regress_w(data):
    y = data[:, 2]
    x = np.zeros((len(data), 2))
    x[:, 0] = data[:, 0]
    x[:, 1] = data[:, 1]
    m = np.linalg.inv(np.dot(np.transpose(x), x))
    n = np.dot(m, np.transpose(x))
    beta_w = np.dot(n, y)
    return beta_w


def regress_x(data):
    y = data[:, 0]
    x = data[:, 1]
    beta_x = float(np.mean(np.multiply(x,y)))/float(np.sum(x**2))
    #m = np.linalg.inv(np.dot(np.transpose(x), x))
    #n = np.dot(m, np.transpose(x))
    #beta_x = np.dot(n, y)
    return beta_x

def loss(data,beta):
    x = data[:, :3]
    y = data[:, 3]
    loss = np.sum((y - np.dot(x, np.transpose(beta)))**2)
    return loss


def constrain(data):
    beta_w = regress_w(data)
    beta_x = regress_x(data)
    k = np.zeros((3, 1))
    k[0] = 1
    k[2] = beta_w[0] + float(beta_w[1]) / float(beta_x + 1)
    k[1] = 1 / float(beta_x + 1)
    return k

def optim(data):
    k = np.transpose(constrain(data))
    x = data[:, :3]
    y = data[:, 3]
    count = 0
    beta = regress_y(data)
    beta = np.zeros((3, 1))
    beta = beta[:, 0]
    gradient = np.zeros((3, 1))
    tol = 1
    c = 1
    lamda_1 = 0.05
    lamda_2 = 0.06
    step = 0.05
    while ((count < 100)): #& (tol > 1e-5)):
        beta_old = beta
        gradient = -2 * np.dot(np.transpose(x), y-np.dot(x, beta)) + lamda_1*k - lamda_2 * k
        beta -= step*gradient[:, 0]
        print beta
        lamda_1 = max(0, lamda_1 + np.dot(k, beta)-c)
        #print lamda_1
        lamda_2 = max(0, lamda_2 - np.dot(k, beta)-c)
        count += 1
        #tol = LA.norm(beta - beta_old)
        #print tol
    return beta



def Effect(data):
    beta_y = regress_y(data)
    beta_w = regress_w(data)
    beta_x = regress_x(data)
    DE = beta_y[0]
    IE = beta_y[2]*beta_w[0]
    SE = float(beta_y[1]+beta_y[2]*beta_w[1])/float(beta_x+1)
    TV = DE + IE + SE
    return DE,IE,SE,TV


def main():
    alpha = 5
    beta1 = 2
    beta2 = 1
    gama1 = 1.5
    gama2 = 0.5
    gama3 = 1
    n = 200
    data = gen_data(alpha, beta1,beta2,gama1,gama2,gama3,n)
    beta_y = regress_y(data)
    #lo = loss(data, beta_y)
    #print lo
    beta_w = regress_w(data)
    beta_x = regress_x(data)
    print constrain(data)
    print Effect(data)
    print beta_y
    #print beta_w
    #print beta_x
    beta = optim(data)
    print "beta", beta

    DE = beta[0]
    IE = beta[2] * beta_w[0]
    SE = float(beta[1] + beta[2] * beta_w[1]) / float(beta_x + 1)
    TV = DE + IE + SE
    print 'TV', TV



    return

main()
