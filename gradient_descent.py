
# coding: utf-8

# <h3>Question 2</h3>

# In[ ]:

import numpy as np
import matplotlib.pyplot as mp


# In[ ]:

########################## Question 2.1 : compute mean squared error ####################
def compute_mse(pred,sample):
      
    totalError = 0
    a = pred
    b = sample 
    
    diff_vals = np.subtract(a,b)
    diff_vals_sq = np.square(diff_vals)
    error_sum = diff_vals_sq.sum()
    mean_sq_error = error_sum / float(diff_vals_sq.size)
    return mean_sq_error


# In[ ]:




# In[59]:

##################################### Q2: 2&3 ###############################
uniform_sample = np.random.uniform(low=0.0,high=1.0,size=100) ##Darw a random sample
sample_size=uniform_sample.size
def calc_y(x,m,c):#function to evaluate y given model and x
    y = m*x+c
    random_x_y = np.vstack((x,y)).T
    return random_x_y

x_y_sample = calc_y(uniform_sample,1,0)
x = x_y_sample[:,0]
y=x_y_sample[:,1]
mp.plot(x,y,color='r') # plot function
#mp.title("y=1*x+0")
mp.show()


# In[21]:

############################ Q2.4 Gradient Descent for y = m*x+c #############################
def perform_batch_SGD_linear(learningRate,m_current,b_current,x,y):
    
    m_gradient=0
    b_gradient=0
    b_gradient += -(2/sample_size) * (y - ((m_current * x) + b_current)) #partial derivative wrt b
    m_gradient += -(2/sample_size) * x * (y - ((m_current * x) + b_current)) #partial derivative wrt m
    b_current = b_current - (learningRate * b_gradient)
    m_current = m_current - (learningRate * m_gradient) 
    
    return [m_current,b_current]
    

def run_gradient_descent(lr):
    m_initial = 0.5
    b_initial = 1
    learningRate=lr
    mse_err=10
    iter=1;
    plot_mse_error=[]
    plot_iter=[]
    predicted = m_initial*x_y_sample[:,0]+b_initial

    while(mse_err>0.00001) and iter<=300 :
        mse_err_old=compute_mse(predicted,x_y_sample[:,1])
        plot_mse_error.append(mse_err_old)
        plot_iter.append(1)
        for i in range(0,99):
            x = x_y_sample[i,0]
            y=x_y_sample[i,1]
            m_initial, b_initial = perform_batch_SGD_linear(learningRate,m_initial,b_initial,x,y)
        predicted = m_initial*x_y_sample[:,0]+b_initial
        mse_err_new = compute_mse(predicted,x_y_sample[:,1])
        mse_err = abs((mse_err_old-mse_err_new)/mse_err_old)
        iter+=1
        plot_iter.append(iter)
        plot_mse_error.append(mse_err_new)
    mp.scatter(plot_iter,plot_mse_error,linewidth=1.0,color='r')
    return plot_iter,plot_mse_error

##########################################################################################


# In[29]:

p1, p2 = run_gradient_descent(0.05)
p0 = list(p1)
px = list(p2)


# In[30]:

p3 , p4 = run_gradient_descent(0.01)
py = list(p3)
pz = list(p4)


# In[32]:

del px[300:]
del pz[300:]
mp.plot(px)
mp.plot(pz)
mp.show()


# In[33]:

######################## Function for y = m1*x+m2*x^2+c ################################

def calc_y_2(x_2,m1,m2,c):
    y_2 = m1*x_2 + m2*x_2**2 + c
    random_x_y = np.vstack((x_2,y_2)).T
    return random_x_y

x_y_sample_2 = calc_y_2(uniform_sample,0.5,1,1)


# In[34]:

############################ Q2.5 repeated for above model #########################
x_axis = np.array(x_y_sample_2[:,0])
y_axis = np.array(x_y_sample_2[:,1])
mp.scatter(x_axis,y_axis,color='r')
mp.show()


# In[35]:

##############################3 Uniform Sample for the model tanh)m*x+c ############################3
uniform_sample = np.random.uniform(low=0.0,high=2.0,size=100)

def calc_y_3(x_3,m,c):
    y_3 = np.tanh(m * x_3 + c)
    random_x_y = np.vstack((x_3,y_3)).T
    return random_x_y

x_y_sample_3 = calc_y_3(uniform_sample,1,2)


# In[36]:

################################## Plot tanh ground truth ########################################################
x_axis = np.array(x_y_sample_3[:,0])
y_axis = np.array(x_y_sample_3[:,1])
mp.scatter(x_axis,y_axis,color='r')
mp.show()


# In[50]:

def perform_batch_SGD_quadratic(learningRate,m1,m2,b,x,y):
    
    dpart = y - (m1 * x+m2*x**2 + b)
    m1_gradient=0
    m2_gradient=0
    b_gradient=0
    b_gradient += -(2/sample_size) * dpart #partial derivative wrt b
    m1_gradient += -(2/sample_size) * x * dpart #partial derivative wrt m1
    m2_gradient+= -(4/x.size) * x * dpart
    b = b - (learningRate * b_gradient)
    m1 = m1 - (learningRate * m1_gradient)
    m2 = m2 - (learningRate * m2_gradient)
    
    return [m1,m2,b]


# In[51]:

def run_gradient_descent_quadratic(lr):
    m1_initial = 0.5
    m2_initial = 0.5
    b_initial = 1
    learningRate=lr
    mse_err=10
    iter=1;
    plot_mse_error=[]
    plot_iter=[]
    x_array = x_y_sample[:,0]
    predicted = m1_initial*x_array+ m2_initial*x_array**2 + b_initial
    while(mse_err>0.00001) and iter<=300:
        mse_err_old=compute_mse(predicted,x_y_sample_2[:,1])
        plot_mse_error.append(mse_err_old)
        plot_iter.append(1)
        for i in range(0,99):
            x = x_y_sample_2[i,0]
            y=x_y_sample_2[i,1]
            m1_initial,m2_initial,b_initial = perform_batch_SGD_quadratic(learningRate,m1_initial,m2_initial,b_initial,x,y)
        predicted = m1_initial*x_array+ m2_initial*x_array**2 + b_initial
        mse_err_new = compute_mse(predicted,x_y_sample_2[:,1])
        mse_err = abs((mse_err_old-mse_err_new)/mse_err_old)
        iter+=1
        plot_iter.append(iter)
        plot_mse_error.append(mse_err_new)
    #mp.scatter(plot_iter,plot_mse_error,linewidth=2.0,color='r')
    return plot_iter,plot_mse_error


# In[54]:

p1, p2 = run_gradient_descent_quadratic(1)
p0 = list(p1)
px = list(p2)

p3 , p4 = run_gradient_descent_quadratic(1.5)
py = list(p3)
pz = list(p4)

del px[300:]
del pz[300:]
mp.plot(px)
mp.plot(pz)
mp.show()


# In[55]:

################################## Run Gradient Descent with model as the tanh function ############
def perform_batch_SGD_tanh(learningRate,m,b,x,y):
    
    m_gradient=0
    b_gradient=0
    dtanh = y - (1-tanh(m *x + b)**2)
    b_gradient += -(2/sample_size) * dtanh #partial derivative wrt b
    m_gradient += -(2/sample_size) * x * dtanh #partial derivative wrt m
    b_current = b_current - (learningRate * b_gradient)
    m_current = m_current - (learningRate * m_gradient) 
    
    return [m,b]
    


# In[56]:

def run_gradient_descent_tanh(lr):
    m_initial = 0.5
    b_initial = 1
    learningRate=lr
    mse_err=10
    iter=1;
    plot_mse_error=[]
    plot_iter=[]
    predicted = m_initial*x_y_sample[:,0]+b_initial

    while(mse_err>0.00001) and iter<=300:
        mse_err_old=compute_mse(predicted,x_y_sample_3[:,1])
        plot_mse_error.append(mse_err_old)
        plot_iter.append(1)
        for i in range(0,99):
            x = x_y_sample_3[i,0]
            y=x_y_sample_3[i,1]
            m_initial, b_initial = perform_batch_SGD_linear(learningRate,m_initial,b_initial,x,y)
        predicted = m_initial*x_y_sample[:,0]+b_initial
        mse_err_new = compute_mse(predicted,x_y_sample_3[:,1])
        mse_err = abs((mse_err_old-mse_err_new)/mse_err_old)
        iter+=1
        plot_iter.append(iter)
        plot_mse_error.append(mse_err_new)
    #mp.plot(plot_iter,plot_mse_error,linewidth=1.0,color='r')
    return plot_iter,plot_mse_error


# In[57]:

p1, p2 = run_gradient_descent_tanh(2)
p0 = list(p1)
px = list(p2)

p3 , p4 = run_gradient_descent_tanh(3)
py = list(p3)
pz = list(p4)

del px[300:]
del pz[300:]
mp.plot(px)
mp.plot(pz)
mp.show()


# In[ ]:



