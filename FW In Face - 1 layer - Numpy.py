
# coding: utf-8

# In[3]:


import numpy as np
from scipy.optimize import linprog



# In[4]:


def sigmoid(u):
    return(1/(1+np.exp(-u)))


# In[5]:


def sigmoid_prime(u):
    return(1/(4*(np.cosh(u/2))**2))


# In[6]:


def mat2vec(A):
    res = np.array([])
    n,p = np.shape(A)
    for i in range(n):
        res = np.concatenate((res,A[i,:]),axis=0)
    return(res)


            
        


# In[7]:


mat2vec(np.identity(3))


# In[8]:


def vec2mat(a,n,p):
    res = np.zeros((n,p))
    for i in range(n):
        res[i,:]=a[(i*p):((i+1)*p)]
    return(res)
    


# In[9]:


vec2mat(np.array([1., 0., 0., 0., 1., 0., 0., 0., 1.,1,2,3]),4,3)


# In[10]:


def lipschitz():
    '''returns the array of lipschitz constants'''
    
    


# In[11]:


def loss(W,X,Y,sigma):
    n,p = np.shape(X)
    _,q = np.shape(Y)
    k1,k0 = np.shape(W)
    res = 0
    for i in range(n):
        loss_i = np.linalg.norm(Y[i,:] - sigma(np.matmul(W,X[i,:])))**2
        res+=loss_i
    return((0.5*res)/n)
    


# In[12]:


#test

Y = np.array([np.matmul(np.ones((3,2)),[10,11]),np.matmul(np.ones((3,2)),[12,13])])
loss(np.ones((3,2)), np.array([[10,11],[12,13]]),Y,lambda x: x)


# In[49]:


def gradient(W,x,y,sigma, sigma_prime):
    
#     print(loss(W,np.array([x]),np.array([y]),sigma))
    
    x = np.array(x)
    y=np.array(y)
    
    k1, k0 = np.shape(W)
    #print(W @ x.reshape(-1,1))
    
    input1 = np.matmul(W,x)
#     print("input1",input1)
    
    u1 = sigma(input1) - y 
#     print("u1",u1)
    
    u2 = np.diag(sigma_prime(input1))
#     print("u2",u2)   
    u3 = np.zeros((k1,k1*k0))

    for i in range(k1):
        u3[i,k0*i:k0*(i+1)] = x
#     print("u3",u3)    
    return(u1 @ u2 @ u3)
    


# In[14]:


#test
gradient(np.ones((2,3)),[4,-2,8],[-1,30],sigmoid,sigmoid_prime)
# np.matmul(np.identity(2),[1,2])
# tested with an extraneous calculator for the linear case and sigmoid case


# In[15]:


def create_matrix_fw(k0,k1,delta,df):
    A_ub = np.zeros((k1,k1*3*k0))
    
     # constraints with delta
    for i in range(k1):
        A_ub[i,(i*3*k0):((i+1)*3*k0)]=np.array([1,1,0]*k0)
    
    b_ub = np.array([delta]*k1)
    
    print(b_ub)
     # non negativity of w+ and w-
    A_ub = np.concatenate((A_ub,np.diag([-1,-1,0]*k0*k1)),axis=0)
    
    print(b_ub.shape,np.zeros((3*k0*k1,)).shape)
    b_ub = np.concatenate((b_ub,np.zeros((3*k0*k1,))),axis=0)
    
    # w = w+ - w-
    A_eq = np.zeros((k1*k0,k1*3*k0))
    for i in range(k0*k1):
        A_eq[i,(i*3):((i+1)*3)]=np.array([1,-1,-1])
        
    b_eq = np.zeros((k0*k1,))
    
    c = np.zeros((3*k0*k1,))
    for i in range(k0*k1):
        c[3*i+2]=df[i]
    return(A_ub,b_ub,A_eq,b_eq,c)
    


# In[16]:


create_matrix_fw(2,3,0.1,np.ones((2*3,))*0.5)


# In[17]:


def create_extremal_points(k0,k1,delta):
    extremal_points = [np.zeros((k0*k1,)) for i in range(2*k0*k1)]
    for i in range(2*k0*k1):
        extremal_points[i][int(i/2)]=(-1)**i*delta
    return(extremal_points)


# In[87]:


def fw_step(W,L,df,delta,Gtilde_1=np.inf,mode="basis"):
    '''
    - mode: basis or linprog
    '''
#     df = gradient()

    w = mat2vec(W)
    
    k1,k0 = np.shape(W)
    if mode == "linprog":
        
        A_ub,b_ub,A_eq,b_eq,c = create_matrix_fw(k0,k1,delta,df)
        print(A_ub,b_ub,A_eq,b_eq,c)
        print(A_ub.shape,b_ub.shape,A_eq.shape,b_eq.shape,c.shape)

        x_tilde=linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=[-np.inf,np.inf])["x"]
        w_tilde = x_tilde[2::3]
    

    
    if mode == "basis":
        extremal_points = create_extremal_points(k0,k1,delta)
        print("extremal_points: ", extremal_points)
        g = lambda x: np.inner(df,x)
        candidate_objectives = list(map(g,extremal_points))
        winner_index = np.argmin(candidate_objectives)
        w_tilde = extremal_points[winner_index]
        print("w_tilde",w_tilde)
        
    #return(w_tilde)
    
#     w_tilde = np.array([X_tilde])
    print("w_tilde",w_tilde)
    #fw gap
    G = df.T @ (w - w_tilde)
    G_tilde = min(G,Gtilde_1)
    
    #curvature Ck
    C = L * np.linalg.norm(w-w_tilde)**2
    
    print("Ck: ",C)
    
    #time step
    alpha = min(G/C,1)
    wiplus1 = w + alpha*(w_tilde-w)
    
    if G <= C:
        theta = G**2/(2*C)
    else:
        theta = G/2
    
    return(vec2mat(wiplus1,k1,k0),G_tilde,theta)


# In[19]:


fw_step(np.zeros((2,1)),1,np.ones((2*1,))*0.5,0.1,mode="linprog")
#tested by comparing with AMPL


# In[20]:


fw_step(np.zeros((2,1)),1,np.ones((2*1,))*0.5,0.1,mode="basis")
#tested by comparing with AMPL


# In[21]:


# k0=2
# k1=3
# extremal_points = [np.zeros((k0*k1,)) for i in range(2*k0*k1)]
# print(extremal_points)
# for i in range(2*k0*k1):
# #     print(i)
#     extremal_points[i][int(i/2)]=(-1)**i*2
# #     print(extremal_points)
# #     print("\n")
    
# u = map(lambda x: np.inner([1,2,3,4,5,6],x),extremal_points)


# In[22]:


# type(u)


# In[23]:


# l= [1,2,3,4,5,6]
# l[2::3]


# In[24]:


#min x st |x| <= 1
# linprog(c=np.array([0,0,1]),A_ub=[[-1,0,0],[0,-1,0],[1,1,0]],b_ub=[0,0,1],A_eq=[[1,-1,-1]],b_eq=[0],bounds = [-np.inf,np.inf])


# In[25]:


# [-1,0,1,0,2].index(0)


# In[26]:


def create_matrix_inface(W,delta,df):
    '''
    creates matrix to have the following LP problem:
    max df * x
    s.t. x in feasible region 
         x in minimal face containing W
    '''
    k1,k0 = np.shape(W)
    
    w = mat2vec(W)
    
    w_is_zero = w==0
    
    num_zero = sum(w_is_zero)
    
    zero_indices = [i for i,x in enumerate(w) if x==0]
#     print(zero_indices)
    
    A_ub,b_ub,A_eq,b_eq,c = create_matrix_fw(k0,k1,delta,df)
    
    inface_A_eq = np.zeros((num_zero,np.shape(A_eq)[1]))
    
    for i in range(num_zero):
        index = zero_indices[i]
        inface_A_eq[i,3*index:3*(index+1)] = [0,0,1]
        
    A_eq = np.concatenate((A_eq, inface_A_eq),axis=0)
    
    b_eq = np.concatenate((b_eq,np.zeros(num_zero,)))
    
    return(A_ub,b_ub,A_eq,b_eq,-c)
    


# In[33]:


def find_inface_dir(W,delta,df, mode = "basis"):
    
    k1,k0 = np.shape(W)
    
    w = mat2vec(W)
    
#     w = mat2vec(W)
    
#     w_is_zero = w==0
    
#     num_zero = sum(w_is_zero)
    
#     zero_indices = [i for i,x in enumerate(w) if x==0]
# #     print(zero_indices)
    
#     A_ub,b_ub,A_eq,b_eq,c = create_matrix_fw(k0,k1,delta,df)
    
#     inface_A_eq = np.zeros((num_zero,np.shape(A_eq)[1]))
    
#     for i in range(num_zero):
#         index = zero_indices[i]
#         inface_A_eq[i,3*index:3*(index+1)] = [0,0,1]
        
#     A_eq = np.concatenate((A_eq, inface_A_eq),axis=0)
    
#     b_eq = np.concatenate((b_eq,np.zeros(num_zero,)))

    if mode == "linprog":
    

        w_hat = linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=[-np.inf,np.inf])["x"]

        w_hat = w_hat[2::3]
#     print(W)
    if mode == "basis":
        extremal_points = create_extremal_points(k0,k1,delta)
        I_plus = (w>0)
        I_zero = (w == 0)
        I_minus = (w<0)
        
        IF_extremal_points = []
        
        #take only in face extremal points
        for p in extremal_points:
            b1 = np.prod(p[I_plus]>=0)
            b2 = np.prod(p[I_zero] == 0)
            b3 = np.prod(p[I_minus] <= 0)
            if b1 + b2 + b3 == 3:
                IF_extremal_points.append(p)
        
        g = lambda x: np.inner(df,x)
        candidate_objectives = list(map(g,IF_extremal_points))
        winner_index = np.argmin(candidate_objectives)
        w_hat = IF_extremal_points[winner_index]
        print(w_hat)
    
    return(vec2mat(w_hat,k1,k0) - W)
    
    
    
    


# In[35]:


#test
# print(find_inface_dir(np.array([[0.25],[0.75],[0]]),1,np.array([1,0,0])))
find_inface_dir(np.array([[0.3],[0.7],[4]]),1,np.array([1,1,0]))
#tested with AMPL


# In[82]:


np.prod([True,False,False])


# In[39]:


def find_alpha_stop(W,delta,df,d,mode="basis"):
    
    k1,k0 = np.shape(W)
    
    if mode == "linprog":
        A_ub,b_ub,A_eq,b_eq,c = create_matrix_inface(W,delta,df)

        print(A_ub.shape,b_ub.shape,A_eq.shape,b_eq.shape,c.shape)

        #add column for variable alpha

        A_ub = np.concatenate((A_ub, np.zeros((np.shape(A_ub)[0],1))),axis=1)
        #b_ub = np.concatenate((b_ub, np.array([0])),axis=0)
        A_eq = np.concatenate((A_eq, np.zeros((np.shape(A_eq)[0],1))),axis=1)
        #b_eq = np.concatenate((b_eq, np.array([0])),axis=0)
        c = np.zeros(np.shape(c))
        c = np.concatenate((c,np.array([-1])),axis=0) # max alpha = min -alpha

        #constraints w - alpha*d = W
        A_line = np.zeros((k0*k1,3*k0*k1))
        for i in range(k0*k1):
            A_line[i,(i*3):((i+1)*3)]=np.array([0,0,1])
        A_line = np.concatenate((A_line, -np.ones((np.shape(A_line)[0],1))),axis=1)

        A_eq = np.concatenate((A_eq,A_line),axis=0)

        b_eq = np.concatenate((b_eq,mat2vec(W)),axis=0)

        print(A_ub.shape,b_ub.shape,A_eq.shape,b_eq.shape,c.shape)

        res = linprog(c=c,A_ub=A_ub,b_ub=b_ub,A_eq=A_eq,b_eq=b_eq,bounds=[-np.inf,np.inf])["x"]
#         print(res)
    
    if mode == "basis":
        res=[1]
    
    return(res[-1])#last variable is alpha
    
    


# In[40]:


find_alpha_stop(np.array([[0.5],[0.5],[0]]),1,np.array([0,1,0]),find_inface_dir(np.array([[0.5],[0.5],[0]]),1,np.array([0,1,0])))
#np.ones((2,1))+alpha_stop*(-2*np.ones((2*1,)))


# In[108]:


def fw_noncvx(X,Y,delta,W,T,gamma_1,gamma_2, activation="linear",sigma=None,sigma_prime=None,k0=None,k1=None):
    '''
    Input:
    X: n x p array 
    Y: n x q array (q dimension of output layer)
    
    
    '''
    
    if(activation == "linear") and sigma is None:
        sigma = lambda x: x
        sigma_prime = lambda x: np.ones(np.shape(x))
        
#     W = np.zeros((k1,k0))
#     i,j = np.random.randint(0,k1-1,1), np.random.randint(0,k0-1,1)
#     W[i,j]=delta
#     L = lipschitz(k0,k1)

    print("Begin Iteration ",0)
    L=10
    n,p = np.shape(X)
    _,q = np.shape(Y)
    
    print(n,p,q)
    
    df = np.zeros((k0*k1,))
    for i in range(n):
        df+=gradient(W,X[i,:],Y[i,:],sigma, sigma_prime)
    df/=n
    
    print("Gradient: ", df)
    
    W, G_tilde, theta = fw_step(W,L,df,delta)
    
    print(W,G_tilde,theta)
    
    #loss function
    f = lambda W: loss(W,X,Y,sigma)
    
    
    
    for k in range(1,T):
        print("Begin Iteration ",k)
        
        #Compute gradient
        df = np.zeros((k0*k1,))
        for i in range(n):
            df+=gradient(W,X[i,:],Y[i,:],sigma, sigma_prime)
        df/=n
        print("Gradient: ", df)
        #Inface direction
        d = find_inface_dir(W,delta,df)
        
        
        #alpha_stop = find_alpha_stop(W,delta,df,d)
        
        alpha_stop = 1
        
        #(a) Go to lower dimensional face
        W_B = W + alpha_stop*d
        beta = np.random.uniform()*alpha_stop
        W_A = W + beta*d
        
        if(f(W_B) <= f(W) - gamma_1*theta):
            W = W_B
            G = G
            G_tilde = G_tilde
            theta = theta
        
        #(b) Stay in current face
        elif(f(W_A) <= f(W) - gamma_2*theta):
            W = W_A
            G = G
            G_tilde = G_tilde
            theta = theta
        
        #(c) Regular FW step
        else:
            print("FW")
            W, G_tilde, theta = fw_step(W,L,df,delta,Gtilde_1=G_tilde)
            print("W: ", W)
    return(W)


# In[42]:


np.random.uniform()


# In[109]:


X = np.array([[1,-2]])
Y = np.array([[0]])
delta = 1
W = np.array([[1,0]])
T=1000
gamma_1 = np.inf
gamma_2 = np.inf


# In[110]:


fw_noncvx(X,Y,delta,W,T,gamma_1,gamma_2,k0=2,k1=1)


# In[91]:


gradient(W,X[0,:],Y[0,:],lambda x:x, lambda x: np.ones(np.shape(x)))


# In[100]:


np.inner([1,-2],[1-0,0-1])


# In[101]:


2*np.linalg.norm([1-0,0-1])**2


# In[103]:


np.array([1,0])+0.75*np.array([0-1,1-0])

