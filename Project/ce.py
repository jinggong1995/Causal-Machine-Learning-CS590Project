import numpy as np
from numpy import linalg as LA
from scipy.stats import bernoulli
import sklearn



def fz(u2):
    return not u2

def fx(u1, z):
    x = (not z and (not u1)) or (z and u1)
    return x

def fw(u3, x, z):
    #w = (not z and not (x) and u3) or (not z and x and u3) or (z and not (x) and not(u3)) or (z and x and not(u3))
    w = ((not x and u3) or(x and not u3)) != z
    return w

def fy(x, z, w, u4):
    #y = (u4 and z) or (not(w) and x) or (not (z) and w) or (not(u4) and w) or (z and not(x))
    y = (w and not u4) != (not x and z)
    return y

def dist(u,p_u):
    p_v = np.zeros((16, 5))
    m = 0
    for i in range(2):
        z = fz(u[1, i])
        for j in range(2):
            x = fx(u[0, j], z)
            for k in range(2):
                w = fw(u[2, k], x, z)
                for l in range(2):
                    y = fy(x, z, w, u[3, l])
                    p_v[m, :] = np.asarray([z, x, w, y, p_u[1, i]*p_u[0, j]*p_u[2, k]*p_u[3, l]])
                    m += 1
    return p_v

def D_effect(p_v, x):
    DE = np.zeros((2, 1))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p_yx1wz = np.sum(p_v[np.sum(p_v[:, :4] == [j, x[1], k, i], axis=1) == 4, 4])
                p_x1wz = np.sum(p_v[np.sum(p_v[:, :3] == [j, x[1], k], axis=1) == 3, 4])
                p_y_x1wz = float(p_yx1wz) / float(p_x1wz)
                p_yx0wz = np.sum(p_v[np.sum(p_v[:, :4] == [j, x[0], k, i], axis=1) == 4, 4])
                p_x0wz = np.sum(p_v[np.sum(p_v[:, :3] == [j, x[0], k], axis=1) == 3, 4])
                p_y_x0wz = float(p_yx0wz) / float(p_x0wz)
                p_x0z = np.sum(p_v[np.sum(p_v[:, :2] == [j, x[0]], axis=1) == 2, 4])
                p_w_x0z = float(p_x0wz) / float(p_x0z)
                p_z_x0 = p_x0z / np.sum(p_v[p_v[:, 1] == x[0], 4])
                DE[i] += (p_y_x1wz - p_y_x0wz) * p_w_x0z * p_z_x0
    return DE[0]

def I_effect(p_v, x):
    IE = np.zeros((2, 1))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p_x1wz = np.sum(p_v[np.sum(p_v[:, :3] == [j, x[1], k], axis=1) == 3, 4])
                p_yx0wz = np.sum(p_v[np.sum(p_v[:, :4] == [j, x[0], k, i], axis=1) == 4, 4])
                p_x0wz = np.sum(p_v[np.sum(p_v[:, :3] == [j, x[0], k], axis=1) == 3, 4])
                p_y_x0wz = float(p_yx0wz) / float(p_x0wz)
                p_x0z = np.sum(p_v[np.sum(p_v[:, :2] == [j, x[0]], axis=1) == 2, 4])
                p_w_x0z = float(p_x0wz) / float(p_x0z)
                p_w_x1z = float(p_x1wz) / float(np.sum(p_v[np.sum(p_v[:, :2] == [j, x[1]], axis=1) == 2, 4]))
                p_z_x1 = float(np.sum(p_v[np.sum(p_v[:, :2] == [j, x[1]], axis=1) == 2, 4])) / float(np.sum(p_v[p_v[:, 1] == x[1], 4]))
                IE[i] += p_y_x0wz * (p_w_x1z - p_w_x0z) * p_z_x1
    return IE[0]


def S_effect(p_v, x):
    SE = np.zeros((2, 1))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                p_yx0wz = np.sum(p_v[np.sum(p_v[:, :4] == [j, x[0], k, i], axis=1) == 4, 4])
                p_x0wz = np.sum(p_v[np.sum(p_v[:, :3] == [j, x[0], k], axis=1) == 3, 4])
                p_y_x0wz = float(p_yx0wz) / float(p_x0wz)
                p_x0z = np.sum(p_v[np.sum(p_v[:, :2] == [j, x[0]], axis=1) == 2, 4])
                p_w_x0z = float(p_x0wz) / float(p_x0z)
                p_z_x1 = float(np.sum(p_v[np.sum(p_v[:, :2] == [j, x[1]], axis=1) == 2, 4])) / float(
                    np.sum(p_v[p_v[:, 1] == x[1], 4]))
                p_x0z = np.sum(p_v[np.sum(p_v[:, :2] == [j, x[0]], axis=1) == 2, 4])
                p_z_x0 = p_x0z / np.sum(p_v[p_v[:, 1] == x[0], 4])
                SE[i] += p_y_x0wz * p_w_x0z * (p_z_x1 - p_z_x0)
    return SE[0]


def learn_LR(data):
    x = data[:, :3]
    class_label = data[:, 3]
    w = np.zeros((101, len(x[1, :])))
    pred_prob = np.zeros((len(data), 1))
    count = 0
    tol = 1
    gradient = np.zeros((len(x[1, :]), 1))
    while ((count < 100) & (tol > 1e-5)):
        for i in range(len(data)):
            pred_prob[i] = 1 / (1 + np.exp(-np.dot(w[count, :], x[i])))
        for j in range(len(gradient)):
            gradient[j] = 0.01 * w[count, j] - sum(np.multiply((class_label - pred_prob.T)[0], x[:, j]))
            w[count + 1, j] = w[count, j] - 0.01 * gradient[j]
        tol = LA.norm(w[count + 1, :] - w[count, :])
        count += 1
    return w[count-1, :]



def test_LR(train, test):
    w = learn_LR(train)
    x = test[:, :3]
    class_label = test[:, 3]
    pred_label = np.zeros((len(test), 1))
    loss = 0
    for i in range(len(test)):
        p1 = 1 / (1 + np.exp(-np.dot(w, x[i])))
        if(p1 >= 0.35):
            pred_label[i] = 1
        if (pred_label[i] != class_label[i]):
            loss += 1
    loss = float(loss)/float(len(test))
    return loss



def gen_data(pz,px,pw,py,n):
    data = np.zeros((n, 4))
    data[:, 0] = bernoulli.rvs(pz, size=n)
    data[:, 1] = bernoulli.rvs(px, size=n)
    data[:, 2] = bernoulli.rvs(pw, size=n)
    data[:, 3] = bernoulli.rvs(py, size=n)
    return data


def new_data_for_w(p_v):
    data = np.zeros((len(p_v), 4))
    data[:, 0] = p_v[:, 0]
    data[:, 1] = p_v[:, 2]
    data[:, 2] = p_v[:, 3]
    data[:, 3] = p_v[:, 2]
    return data


def new_data_for_x(p_v):
    data = np.zeros((len(p_v), 4))
    data[:, 0] = p_v[:, 0]
    data[:, 1] = p_v[:, 3]
    data[:, 2] = p_v[:, 2]
    data[:, 3] = p_v[:, 1]
    return data


def main(u, p_u):
    p_v = dist(u,p_u)
    print p_v

    x = [0, 1]
    DE = D_effect(p_v, x)
    IE = I_effect(p_v, x)
    SE = S_effect(p_v, x)
    print DE
    print IE
    print SE



    px = 0.7
    py = 0.4
    pz = 0.6
    pw = 0.55
    n = 50
    data = gen_data(pz, px, pw, py, n)
    train = p_v
    test = p_v
    cof_y = learn_LR(train)
    print cof_y
    loss = test_LR(train, test)
    print loss
    data_w = new_data_for_w(p_v)
    cof_w = learn_LR(data_w)
    print cof_w
    data_x = new_data_for_x(p_v)
    cof_x = learn_LR(data_x)
    print cof_x


    return



u = np.zeros((4, 2))
u[:, 0] = 1
p_u = np.zeros((4, 2))
p_u[0, :] = [0.55, 0.45]
p_u[1, :] = [0.4, 0.6]
p_u[2, :] = [0.4, 0.6]
p_u[3, :] = [0.45, 0.55]
main(u, p_u)




