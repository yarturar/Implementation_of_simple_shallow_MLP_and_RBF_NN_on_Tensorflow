import numpy as np
import tensorflow as tf
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from math import sqrt
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

def MLP(data, n_hidden, rho, lr):
    
    n_input = 2
    n_output = 1
    xd=np.array(data.iloc[:, :2])
    yd=np.array(data.iloc[:, 2])
    yd=yd.reshape((200,1))
    X_train, X_test, y_train, y_test = train_test_split(xd, yd, test_size=0.25, random_state=1820436)
    
    weights = {'w': tf.Variable(tf.random_uniform(shape=[n_input, n_hidden],
                                                  minval=-2, maxval=3, dtype=tf.float32, seed=36), name='w'),
               'v': tf.Variable(tf.random_uniform(shape=[n_hidden, n_output], minval=-1, maxval=4,
                                                  dtype=tf.float32, seed=36), name='v')}
    bias = {'b': tf.Variable(tf.random_uniform(shape=[n_hidden], minval=-5, maxval=4,
                                               dtype=tf.float32, seed=36), name='b')}

    x=tf.placeholder(tf.float32, [None,n_input])
    y=tf.placeholder(tf.float32, [None, n_output])

    layer1 = tf.subtract(tf.matmul(x, weights['w']), bias['b'])
    layer1_act = tf.tanh(layer1 / 2)
    out_l = tf.matmul(layer1_act, weights['v'])
    cost = tf.reduce_mean(tf.squared_difference(y, out_l))
    reg_cost = cost + rho * (tf.nn.l2_loss(weights['w']) + tf.nn.l2_loss(weights['v']) + tf.nn.l2_loss(bias['b']))
    
    grad = tf.gradients(reg_cost, [weights['w'], weights['v'], bias['b']])
    
    train_w_b=tf.train.AdamOptimizer(learning_rate=lr).minimize(reg_cost, var_list=[weights['w'], bias['b']])
    train_v=tf.train.AdamOptimizer(learning_rate=lr).minimize(reg_cost, var_list=[weights['v']])

    start_time = time.time()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(4000):
            w_b, loss_val_w_b= sess.run([train_w_b, reg_cost], 
                 feed_dict={x:X_train, y:y_train})
            v, loss_val_v= sess.run([train_v, reg_cost], 
                 feed_dict={x:X_train, y:y_train})
            if epoch == 0:
                pred_initial = sess.run(out_l,feed_dict={x:X_train})
        pred_y_test = sess.run(out_l, feed_dict={x: X_test})
        pred_y_train = sess.run(out_l, feed_dict={x: X_train})

        grad_opt = sess.run(grad, feed_dict={x: X_train, y: y_train})
        
        fnormgrad=[*grad_opt]
        fnormgrad=np.array(fnormgrad)
        arrng=fnormgrad[fnormgrad != np.array(None)]
        eng=np.concatenate( arrng, axis=0 )
        hz=(sum(eng[0]**2)+sum(eng[1]**2)+sum(eng[2]**2))**(0.5)
        norm_grad_opt = np.sqrt(np.sum(hz))
        
        x1=np.linspace(-2, 2 ,50)
        x2=np.linspace(-2,2,50)
        X_1, X_2=np.meshgrid(x1,x2)
        XX=np.vstack([ X_1.reshape(-1), X_2.reshape(-1) ]).T
        ygen= sess.run(out_l, feed_dict={x:XX})

    end_time = time.time()
    
    mse_initial = mean_squared_error(y_train, pred_initial)/2
    mse_train = mean_squared_error(y_train,pred_y_train)/2
    mse_test = mean_squared_error(y_test, pred_y_test)/2
    
    def fun_plot(Z): 
    
        x1=np.linspace(-2, 2 ,50)
        x2=np.linspace(-2,2,50)
        X1, X2=np.meshgrid(x1,x2)
        Z=np.reshape(Z, (50,50))
        plt.figure(figsize=(20,10))

        ax = plt.axes(projection='3d')
        ax.plot_surface(X1, X2, Z, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(40, 305)

        return(plt.show())
    
    
    print('Number of neurons N: ', n_hidden)
    print('Initial Training Error: ', mse_initial)
    print('Final Train Error: ', mse_train)
    print('Final Test Error: ', mse_test)
    print('Optimization solver chosen: AdamOptimizer')
    print('Norm of the gradient at the optimal point: ',norm_grad_opt )
    print('Time for optimizing the network: %s seconds' % round(end_time - start_time))
    print('value of sigma: 1')
    print('value of rho: 0.00001')
    print('Other hyperparameters:(number of epochs)): 4000')
    
    return (fun_plot(ygen))



    
    
    
    
    
    
    
    
    
    
    
    
    