
# coding: utf-8

# In[2]:


import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import scipy
from mpl_toolkits import mplot3d
from pylab import *
import random
import time

def rbf(data, n_hidden, rho, sigma): 
    
    xd=np.array(data.iloc[:, :2])
    yd=np.array(data.iloc[:, 2])
    yd=yd.reshape((200,1))
    X_train, X_test, y_train, y_test = train_test_split(xd, yd, test_size=0.25, random_state=1820436)
    
    n_input=2
    n_output=1
    
    centers = {'c': tf.Variable(tf.random_uniform(shape=[ n_input, n_hidden], 
                                                 minval=-.5, maxval=.5, dtype=tf.float32, seed=1820436), name='c')}
    weights = { 'v' : tf.Variable(tf.random_uniform(shape=[n_hidden, n_output], minval=-2, maxval=0.5,
                                                dtype=tf.float32, seed=1820436), name='v')}

    x=tf.placeholder(tf.float32, [None,n_input])
    y=tf.placeholder(tf.float32, [None, n_output])

    #first layer
    x_2=tf.reduce_sum(tf.square(x), 1)
    c_2=tf.reduce_sum(tf.square(centers['c']), 0)
    x2upd=tf.reshape(x_2, [-1, 1])
    c2upd=tf.reshape(c_2, [1, -1])
    
    r=x2upd-2*tf.matmul(x, centers['c'])+c2upd
    gauss=tf.exp(-(r/sigma**2))

    #output layer
    out_l=tf.matmul(gauss, weights['v'])

    cost = tf.reduce_mean(tf.squared_difference(y, out_l))/2
    reg_cost=cost+rho*(tf.nn.l2_loss(centers['c'])+tf.nn.l2_loss(weights['v']))

    
    
    
    grad = tf.gradients(reg_cost, [centers['c'], weights['v']]) # this for gettting norm of gradient at optimal point
      
    #scipy optmimzer
    scipy_opt=tf.contrib.opt.ScipyOptimizerInterface(reg_cost,  method='L-BFGS-B',
                                                    options={'maxiter': 2000, 'ftol':1e-10})
    
    
    
    train_error = []
    test_error = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #initial train error
        tie= sess.run(out_l, feed_dict={x:X_train})
        train_init_error=np.square(np.subtract(tie, y_train)).mean()/2
            
        #minimization
        grad_opt1 = sess.run(grad, feed_dict={x: X_train, y: y_train})  
        random.seed(1820436)
        
        start_time = time.time()
        
        scipy_opt.minimize(session=sess, feed_dict={x: X_train, y: y_train}) 
        
        elapsed_time=time.time()-start_time
        
        
        pred_y = sess.run(out_l, feed_dict={x:X_test})
        
        grad_opt = sess.run(grad, feed_dict={x: X_train, y: y_train})
        #norm_grad_opt
        fnormgrad=[*grad_opt]
        fnormgrad=np.array(fnormgrad)
        arrng=fnormgrad[fnormgrad != np.array(None)]
        eng=np.concatenate( arrng, axis=0 )
        norm_grad_opt=(sum(eng[0]**2)+sum(eng[1]**2))**(0.5)
        
        ##################
        
        #final train error
        
        tfe= sess.run(out_l, feed_dict={x:X_train})
        train_final_error=np.square(np.subtract(tfe, y_train)).mean()/2
        
    #building approximating function
        x1=np.linspace(-2, 2 ,50)
        x2=np.linspace(-2,2,50)
        X_1, X_2=np.meshgrid(x1,x2)
        XX=np.vstack([ X_1.reshape(-1), X_2.reshape(-1) ]).T
        ygen= sess.run(out_l, feed_dict={x:XX})
    
    #final test error
       
        error=np.square(np.subtract(pred_y, y_test)).mean()/2
    
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
    
    
    return (print('\n Nunmber of neurons:', n_hidden),
           print('Initial Training Error:', train_init_error),
           print('Final Training Error:', train_final_error),
           print('Final Test Error:', error),
           print('Optimization Solver Chosen: L-BFGS-B'),
           print('Norm of Gradient at the optimal point:', norm_grad_opt ),
           print('Time for optimizing the network(sec):', elapsed_time ),
           print('value of sigma:', sigma),
           print('value of rho:', rho),
           print('Approximating function plot: \n'),
            fun_plot(ygen))

