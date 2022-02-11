# BIC F20 Assignment 2 Problem 1
# Written by Tristan Kilper

import numpy as np

# Sample feature set + labels
feature_set = np.array([[66.5,66.5],[67,66.5],[66.5,67],[67,67]])
labels1 = np.array([66.5,66.5,66.5,67])
labels1 = labels1.reshape(4,1)

# random seed
np.random.seed(42)

# input neuron x

def in_x(Ix, t):
    Cm = 1.6
    Rm = 1.5
    input_I = Ix
    Vm = 2
    thold = 0
    Vr = -65

    Vx = 66

    def out_V(Vcur, t):
        return ((-1 * (Vcur - Vr) + input_I*t)/Cm) - (Vm*t)/(Rm*Cm)

    time = t # ms
    cur_t = 0
    cur_out = Vr

    while cur_t <= time:
        print('t: ' + str(cur_t) + '   V: ' + str(cur_out))
        cur_out += out_V(cur_out, 1)
        if cur_out >= thold:
            print('[spike!]')
            cur_out = Vr
            Vx += .5
        cur_t += 1

    return Vx

# input neuron y

def in_y(Iy, t):
    Cm = 1.6
    Rm = 1.5
    input_I = Iy
    Vm = 2
    thold = 0
    Vr = -65

    Vy = 66

    def out_V(Vcur, t):
        return ((-1 * (Vcur - Vr) + input_I*t)/Cm) - (Vm*t)/(Rm*Cm)

    time = t # ms
    cur_t = 0
    cur_out = Vr

    while cur_t <= time:
        print('t: ' + str(cur_t) + '   V: ' + str(cur_out))
        cur_out += out_V(cur_out, 1)
        if cur_out >= thold:
            print('[spike!]')
            cur_out = Vr
            Vy += .5
        cur_t += 1

    return Vy

# output neuron z

def out_z(Iz, t):
    Cm = 1.6
    Rm = 1.5
    input_I = Iz
    Vm = 2
    thold = 0
    Vr = -65

    Vz = 66

    def out_V(Vcur, t):
        return ((-1 * (Vcur - Vr) + input_I*t)/Cm) - (Vm*t)/(Rm*Cm)

    time = t # ms
    cur_t = 0
    cur_out = Vr

    while cur_t <= time:
        print('t: ' + str(cur_t) + '   V: ' + str(cur_out))
        cur_out += out_V(cur_out, 1)
        if cur_out >= thold:
            print('[spike!]')
            cur_out = Vr
            Vz += .5
        cur_t += 1

    return Vz

# learning rules
a1 = 0.0000005
a2 = 0.0000001
a3 = 0.0000001


def d_w1(x,z):
    return (a1*x*z) - (a2*x) - (a3*z)


b1 = 0.0000005
b2 = 0.0000001
b3 = 0.0000001


def d_w2(y,z):
    return (b1*y*z) - (b2*y) - (b3*z)


# translate spike train to inputs
def readSpikeTrain(featset):

    x_ins = np.array([-1,-1,-1,-1])
    y_ins = np.array([-1,-1,-1,-1])


    for n in range(4):
        x_ins[n] = in_x(featset[n,0], 10)
        y_ins[n] = in_y(featset[n,1], 10)

    # x_ins = x_ins.reshape(4,1)
    # y_ins = y_ins.reshape(4,1)

    inputs = np.array([x_ins, y_ins])
    inputs = inputs.T

    return inputs

# train network
def train_network(inputs, labels):
    weights = np.random.rand(2,1)
    weights /= 2.55
    print(inputs)

    for epochs in range(100):
        print(weights)

        # output value
        Vz_hat = []
        Iz_hat = np.dot(inputs, weights)
        print(Iz_hat)
        for I in Iz_hat:
            Vz_hat.append(out_z(I, 10))
        print(Vz_hat)

        for num in range(4):
            weights[0] += d_w1(inputs[num,0], Vz_hat[num])
            weights[1] += d_w2(inputs[num,1], Vz_hat[num])
            if weights[0] > .6125 or weights[1] > .6125:
                break

        if weights[0] > .6125 or weights[1] > .6125:
            print(weights)
            break

    return weights

test_inputs = readSpikeTrain(feature_set)
fin_weights = train_network(test_inputs, labels1)

# final test for all cases
single_point1 = np.array([66.5, 67])
single_point2 = np.array([66.5, 66.5])
single_point3 = np.array([67, 66.5])
single_point4 = np.array([67, 67])

result1 = out_z(np.dot(single_point1, fin_weights), 10)
result2 = out_z(np.dot(single_point2, fin_weights), 10)
result3 = out_z(np.dot(single_point3, fin_weights), 10)
result4 = out_z(np.dot(single_point4, fin_weights), 10)

print("Final Weights:")
print(fin_weights)

print("Final Test Cases:")
print(single_point1)
print(result1)
print(single_point2)
print(result2)
print(single_point3)
print(result3)
print(single_point4)
print(result4)
