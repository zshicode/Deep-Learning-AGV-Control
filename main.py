import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
import lstm
import transformer
import time
import torch

epochs = 200
lr = 0.01
wd = 1e-5
random_state = np.random.RandomState(42)
torch.manual_seed(42)
T = 0.01
step = 200
model = 'T'
# T: Transformer, L: LSTM, TL: Transformer-LSTM, else: EM-KF
coord = 'C'
# C: cartesian (3-DOF), P: polar (2-DOF)
scene = 'K'
# K: kinematics, M: mechanics
vc = 1
xcoord0 = 1
ycoord0 = 1
theta0 = np.pi/12

def lqr(A,B,Q,R):
    RI = np.linalg.inv(R)
    H = np.vstack((np.hstack((A,-np.dot(np.dot(B,RI),B.T))),
    np.hstack((-Q,-A.T))))
    x,d = np.linalg.eig(H)
    j = 0
    V2 = np.array(np.zeros(A.shape))
    V1 = np.array(np.zeros(A.shape))
    n = A.shape[0]
    nd = d.shape[0]
    d = d.T

    for i in range(np.linalg.matrix_rank(H)):
        if x[i].real < 0:
            V1[j,:] = d[i,0:n]
            V2[j,:] = d[i,n:nd]
            j = j+1
    
    P = np.mat(np.dot(V2,np.linalg.pinv(V1)))
    K = np.dot(np.dot(RI,B.T),P)
    return K

def pseudolqr(A,B,Q,R):
    RI = np.linalg.inv(R)
    H = np.vstack((np.hstack((np.dot(np.dot(B,RI),B.T),-A)),
    np.hstack((-A.T,-Q))))
    _,PI = np.linalg.eig(H)
    P = PI[:A.shape[0],:A.shape[0]]
    K = np.dot(np.dot(RI,B.T),P)
    return K

'''
Step 1: Provide actual system
'''
if scene == 'M':
    pole = 0.1
    cart = 1.0
    g = 9.8
    L = 0.5
    inertia = 1.33*pole*L*L
    force = 1
    ptr = inertia*(cart+pole)+pole*cart*L*L
    A = np.array([[0,1,0,0],
            [ 0, 0, pole*pole*g*L*L/ptr, 0],
            [ 0, 0, 0, 1],
            [0, 0, pole*g*L*(pole+cart)/ptr, 0]])
    B = np.array([[0], [(inertia+pole*L*L)*force/ptr], [0], [pole*L*force/ptr]])
    m0 = np.array([xcoord0,vc,theta0,0])
    C = np.array([1,0,0,0])
    D = 0
    B0 = np.array([0,0,0,0])
else:
    if coord == 'C':
        A = np.array([[0,0,0],
            [ 0, 0, vc],
            [ 0, 0, 0]])
        B = np.array([[0], [0], [1]])
        m0 = np.array([xcoord0,ycoord0,theta0])
        C = np.array([np.cos(theta0),np.sin(theta0),0])
        D = np.linalg.norm(m0[:2])-np.inner([np.cos(theta0),np.sin(theta0)],m0[:2])
        B0 = np.array([T,0,0])
    else:
        A = np.array([[0,vc],
            [ 0, 0]])
        B = np.array([[0], [1]])
        m0 = np.array([0,theta0])
        C = np.array([1,0])
        D = 0
        B0 = np.array([0,0])

sigma2 = np.power(0.03,2)
num_state = B.shape[0]
Q = sigma2*np.eye(num_state)
R = sigma2*np.eye(1)
P0 = sigma2*np.eye(num_state)
K = pseudolqr(A,B,Q,R)
A0 = A - B.dot(K)
A0 = np.eye(num_state) + T*A0
RI = np.linalg.inv(R)
KP = np.dot(np.dot(RI,B.T),P0)
AP = A - B.dot(KP)
AP = np.eye(num_state) + T*AP

kft = KalmanFilter(
    A0,C,Q,R,B0,D,m0,P0,
    random_state=random_state
)# model should be
state, observation = kft.sample(
    n_timesteps=step,
    initial_state=m0
)# provide data

'''
Step 2: Initialize our model
'''

# sample from model

kf = KalmanFilter(
    AP,C,Q,R,B0,D,m0,P0,
    random_state=random_state,
    em_vars=['transition_matrices']
)
data = kf.sample(n_timesteps=step,initial_state=m0)[1]
filtered_state_estimater, nf_cov = kf.filter(observation)
smoothed_state_estimater, ns_cov = kf.smooth(observation)

'''
Step 3: Learn good values for parameters named in `em_vars` using the EM algorithm
'''

def compute_tr(a):
    size = a.shape[0]
    return (np.trace(a)/size)*np.eye(size)

def test(data,method='TL',n_iteration=10):
    t_start = time.process_time()
    if method == 'TL':
        print('----transformer+lstm----')
        data,loss_list = transformer.train(data,step,epochs,lr,wd)
        data,loss_list = lstm.train(data,step,epochs,lr,wd)
        labelfilter = 'TL-KF'
        labelsmooth = 'TL-KS'
    elif method == 'L':
        print('----lstm----')
        data,loss_list = lstm.train(data,step,epochs,lr,wd)
        labelfilter = 'LSTM-KF'
        labelsmooth = 'LSTM-KS'
    elif method == 'T':
        print('----transformer----')
        data,loss_list = transformer.train(data,step,epochs,lr,wd)
        labelfilter = 'Transformer-KF'
        labelsmooth = 'Transformer-KS'
    else:
        print('----EM----')
        labelfilter = 'EM-KF'
        labelsmooth = 'EM-KS'
    
    t_train = time.process_time()
    kfem = kf.em(X=data, n_iter=n_iteration)
    t_em = time.process_time()
    print('train-time/sec',t_train-t_start)
    print('em-time/sec',t_em-t_train)
    Aem = kfem.transition_matrices
    print('Aem=',Aem)
    kfem = KalmanFilter(
        Aem,C,Q,R,B0,D,m0,P0,
        random_state=random_state
    )
    #obsem = kfem.sample(n_timesteps=step,initial_state=m0)[1]
    filtered_state_estimates, f_cov = kfem.filter(observation)
    smoothed_state_estimates, s_cov = kfem.smooth(observation)
    return filtered_state_estimates, f_cov, smoothed_state_estimates, s_cov,labelfilter,labelsmooth


# draw estimates
filtered_state_estimates, f_cov, smoothed_state_estimates, s_cov, labelfilter,labelsmooth = test(data[:,0],method=model,n_iteration=5)
filtered_delta_estimater = filtered_state_estimater[:,0] - state[:,0]
smoothed_delta_estimater = smoothed_state_estimater[:,0] - state[:,0]
filtered_delta_estimates = filtered_state_estimates[:,0] - state[:,0]
smoothed_delta_estimates = smoothed_state_estimates[:,0] - state[:,0]
maefr = np.abs(filtered_delta_estimater).mean()
maesr = np.abs(smoothed_delta_estimater).mean()
maefs = np.abs(filtered_delta_estimates).mean()
maess = np.abs(smoothed_delta_estimates).mean()
print('----MAE----')
print('KF',maefr)
print('KS',maesr)
print(labelfilter,maefs)
print(labelsmooth,maess)

#draw
taxis = np.linspace(0,step*T,step)
plt.figure()
lines_filter = plt.scatter(taxis,state[:,0], color='c',label='True')
lines_filter = plt.plot(taxis,filtered_state_estimater[:,0], 'r',label='KF')
lines_smoother = plt.plot(taxis,smoothed_state_estimater[:,0], 'r--',label='KS')
lines_filt = plt.plot(taxis,filtered_state_estimates[:,0], 'b',label=labelfilter)
lines_smooth = plt.plot(taxis,smoothed_state_estimates[:,0], 'b--',label=labelsmooth)
plt.xlim(0,step*T)
plt.xlabel('Time/s')
plt.ylabel('x/m')
plt.legend()
plt.grid()
plt.figure()
dlines_filter = plt.plot(taxis,filtered_delta_estimater, 'r',label='KF')
dlines_smoother = plt.plot(taxis,smoothed_delta_estimater, 'r--',label='KS')
dlines_filt = plt.plot(taxis,filtered_delta_estimates, 'b',label=labelfilter)
dlines_smooth = plt.plot(taxis,smoothed_delta_estimates, 'b--',label=labelsmooth)
plt.xlim(0,step*T)
plt.xlabel('Time/s')
plt.ylabel('Error/m')
plt.legend()
plt.grid()
plt.show()

def path(state,x0,y0,theta0):
    if coord == 'C':
        return state[:,0],state[:,1],state[:,2]
    else:
        xl = []
        yl = []
        tl = []
        x = x0
        y = y0
        theta = theta0
        for i in range(len(state)):
            xl.append(x)
            yl.append(y)
            tl.append(theta)
            x = x0 + vc*T*i*np.cos(theta)
            y = y0 + state[i,0]
            theta = state[i,1]
        
        return xl,yl,tl

if scene != 'M':
    # draw path for agv
    plt.figure()
    tx,ty,tt = path(state,xcoord0,ycoord0,theta0)
    fx,fy,ft = path(filtered_state_estimater,xcoord0,ycoord0,theta0)
    sx,sy,st = path(smoothed_state_estimater,xcoord0,ycoord0,theta0)
    fsx,fsy,fst = path(filtered_state_estimates,xcoord0,ycoord0,theta0)
    ssx,ssy,sst = path(smoothed_state_estimates,xcoord0,ycoord0,theta0)
    plt.plot(tx,ty, color='c',label='True')
    plt.plot(fx,fy, 'r',label='KF')
    plt.plot(sx,sy, 'r--',label='KS')
    plt.plot(fsx,fsy, 'b',label=labelfilter)
    plt.plot(ssx,ssy, 'b--',label=labelsmooth)
    plt.xlabel(r'$x_1$/m')
    plt.ylabel(r'$x_2$/m')
    plt.legend()
    plt.grid()
    plt.show()