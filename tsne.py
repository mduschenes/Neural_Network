#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np

import sys
sys.path.insert(0, "C:/Users/Matt/Google Drive/PSI/"+
				   "PSI Essay/PSI Essay Python Code/tSNE_Potts/")

from data_functions import Data_Process 
from misc_functions import dict_modify,display


# Compute Squared Pairwise distance between all elements in x
def pairwise_distance(x):
	x2 =  np.dot(x,x.T) 
	x2_d = np.diagonal(x2)
	return x2_d + (x2_d - 2*x2).T
#    m = np.shape(x)[0]
#    return  np.tile(x2_d,(m,1)).T + np.tile(x2_d,(m,1)) -2*(x2) 

# Compute Pairwise distance between all elements in x, exluding self-distances
def neighbour_distance(x,n):
	return pairwise_distance(x)[~np.eye(n,dtype=bool)].reshape(n,n-1)

# Compute guassian representation of pairwise distances
def rep_gaussian(d,sigma,normalized=True,tol=1e-14):
	p = np.atleast_2d(np.exp(-d*sigma))
	np.fill_diagonal(p,0)
	if normalized: return p/np.sum(p,axis=1)
	else: return p,np.maximum(np.sum(p,axis=1),tol)

def rep_tdistribution(d,sigma,normalized=True,tol=1e-14):
	p = np.power(1+d,-1)
	np.fill_diagonal(p,0)
	if normalized: return p/np.sum(p,axis=1)
	else: return p,np.maximum(np.sum(p,axis=1),tol)
	

def entropy(p,q,axis=-1):
	return np.sum(q*np.log(q),axis=axis)


def entropy_gaussian(d,sigma):
	p,norm_p = rep_gaussian(d,sigma,normalized=False)
	return np.power(norm_p,-1)*sigma*np.sum(p*d,1),p


def KL_entropy(p,q):
	return entropy(p,p) - entropy(p,q)


def binary_search(x,a,f,f0,tol,n_iter):
		
		fi,y = f(x,a)        
		n = np.shape(y)[0]
		df = fi-f0
		
		i = 0
		a_min = -np.inf*np.ones(n)
		a_max = np.inf*np.ones(n)
		
		
		while np.any(np.abs(df) > tol) and i < n_iter:
			ind_pos = df>0 
			ind_neg = df<=0
			
			ind_min_inf = np.abs(a_min)==np.inf
			ind_max_inf = np.abs(a_max)==np.inf
			
			#print('f',i,fi)
			
			ind_min_ninf = np.logical_not(ind_max_inf)
			ind_max_ninf = np.logical_not(ind_min_inf)
			
			a_min[ind_pos] = a[ind_pos].copy()
			a_min[ind_neg] = a[ind_neg].copy()

			a[ind_pos & ind_max_inf] *=4
			a[ind_pos & ind_max_ninf] += a_max[ind_pos & ind_max_ninf]
			a[ind_neg & ind_min_ninf] += a_min[ind_neg & ind_min_ninf]
			a /= 2
			
			#print(a)

			fi,y = f(x,a)
			df = fi-f0
			i += 1
			
		print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / a)))
		print("Mean value of Perp: %f" % np.mean(fi))

		return y











def Hbeta(D=np.array([]), beta=1.0):
	"""
		Compute the perplexity and the P-row for a specific value of the
		precision of a Gaussian distribution.
	"""

	# Compute P-row and corresponding perplexity
	P = np.exp(-D.copy() * beta)
	sumP = np.maximum(np.sum(P,axis=-1),10**(-14))
	H = np.log(sumP) + beta * np.sum(D * P) / sumP
	P = P / sumP
	return H, P


def x2p(X=np.array([]),f=Hbeta, tol=1e-5, perplexity=30.0,n_iter=50):
	"""
		Performs a binary search to get P-values in such a way that each
		conditional Gaussian has the same perplexity.
	"""

	# Initialize some variables
	print("Computing pairwise distances...")
	(n, _) = X.shape
#    sum_X = np.sum(np.square(X), 1)
#    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
	P = np.zeros((n, n))
	beta = np.ones((n, 1))

	# Loop over all datapoints
	for i in range(n):

#        # Print progress
#        if i % 500 == 0:
#            print("Computing P-values for point %d of %d..." % (i, n))

		# Compute the Gaussian kernel and entropy for the current precision
		betamin = -np.inf
		betamax = np.inf
		Di = X[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
		(H, thisP) = f(Di, beta[i])

		# Evaluate whether the perplexity is within tolerance
		Hdiff = H - perplexity
		tries = 0
		while np.abs(Hdiff) > tol and tries < 50:

			# If not, increase or decrease precision
			if Hdiff > 0:
				betamin = beta[i].copy()
				if betamax == np.inf or betamax == -np.inf:
					beta[i] = beta[i] * 2.
				else:
					beta[i] = (beta[i] + betamax) / 2.
			else:
				betamax = beta[i].copy()
				if betamin == np.inf or betamin == -np.inf:
					beta[i] = beta[i] / 2.
				else:
					beta[i] = (beta[i] + betamin) / 2.
			# print(beta[i])
			# Recompute the values
			(H, thisP) = Hbeta(Di, beta[i])
			Hdiff = H - perplexity
			tries += 1

		# Set the final row of P
		P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

	# Return final P-matrix
	print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
	print("Mean value of Perp: %f" % np.mean(H))

	return P


def pca(X,n_dims=None):    
	
	if n_dims is None:
		n_dims = np.shape(X)[1]
	
	Xc = X - np.mean(X,0)
	
	L,P = np.linalg.eig(np.dot(Xc.T,Xc))
	
	return np.dot(Xc,P[:, 0:n_dims]), L/np.sum(L),P


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
	"""
		Runs t-SNE on the dataset in the NxD array X to reduce its
		dimensionality to no_dims dimensions. The syntaxis of the function is
		`Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
	"""

	# Check inputs
	if isinstance(no_dims, float):
		print("Error: array X should have type float.")
		return -1
	if round(no_dims) != no_dims:
		print("Error: number of dimensions should be an integer.")
		return -1

	# Initialize variables
	X,_,_ = pca(X, initial_dims)
	(n, d) = X.shape
	max_iter = 1000
	initial_momentum = 0.5
	final_momentum = 0.8
	eta = 500
	min_gain = 0.01
	Y = np.random.randn(n, no_dims)
	dY = np.zeros((n, no_dims))
	iY = np.zeros((n, no_dims))
	gains = np.ones((n, no_dims))

	# Compute P-values
	P = x2p(X, 1e-5, perplexity)
	P = P + np.transpose(P)
	P = P / np.sum(P)
	P = P * 4.									# early exaggeration
	P = np.maximum(P, 1e-12)

	# Run iterations
	display(True,True,'Iterations for t-SNE')
	
	for i in range(max_iter):

		# Compute pairwise affinities
		sum_Y = np.sum(np.square(Y), 1)
		num = -2. * np.dot(Y, Y.T)
		num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
		num[range(n), range(n)] = 0.
		Q = num / np.sum(num)
		Q = np.maximum(Q, 1e-12)

		# Compute gradient
		PQ = P - Q
		for nn in range(n):
			dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

		# Perform the update
		if i < 20:
			momentum = initial_momentum
		else:
			momentum = final_momentum
		gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
				(gains * 0.8) * ((dY > 0.) == (iY > 0.))
		gains[gains < min_gain] = min_gain
		iY = momentum * iY - eta * (gains * dY)
		Y = Y + iY
		Y = Y - np.tile(np.mean(Y, 0), (n, 1))

		# Compute current value of cost function
		if (i + 1) % max_iter/5 == 0:
			C = np.sum(P * np.log(P / Q))
			display(m="Iteration %d: error is %f" % (i + 1, C))

		# Stop lying about P-values
		if i == max_iter/10:
			P = P / 4.

	# Return solution
	return Y
	
def Hbeta0(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = np.maximum(sum(P),10e-12)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p0(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta0(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta0(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca0(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne0(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca0(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p0(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y




		
def plot_props(keys,data_rep,data_type):
	return {
		 k: {
		
		  'ax':   {'title' : '', 
					'xlabel': 'x1', 
					'ylabel': 'x2'},
		  
		  'plot':  {'c': np.reshape(data_typed[(
							 'temperatures'+'_'+data_type)][k],(-1,))},
		  
		  'data':  {'plot_type':'scatter',
					'plot_range': np.reshape(data_typed[(
							 'temperatures'+'_'+data_type)][k],(-1,)),
					'data_process':lambda data: np.real(data)},
					
		  'other': {'cbar_plot':True, 'cbar_title':'Temperatures',
				   'cbar_color':'jet','cbar_color_bad':'magenta',
					'label': lambda x='':x,'pause':0.01,
					'sup_title': {'t': data_rep + ' Representation - ' +
									   data_type}
					}
		 }
		for k in keys}



if __name__ == "__main__":
	pass
#    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
#    print("Running example on 2,500 MNIST digits...")
#    X = np.loadtxt("mnist2500_X.txt")
#    labels = np.loadtxt("mnist2500_labels.txt")
#
#    
#    
#    
	# Import Data
	temperatures_Ising = ['temperatures_Ising_L20', 'temperatures_Ising_L40',
						  'temperatures_Ising_L80']
	
	temperatures_gauge = ['temperatures_gaugeTheory_L20', 
						  'temperatures_gaugeTheory_L40',
						  'temperatures_gaugeTheory_L80']
	
	ising = ['spinConfigs_Ising_L20','spinConfigs_Ising_L40',
		     'spinConfigs_Ising_L80']
	
	gauge = ['spinConfigs_gaugeTheory_L20', 'spinConfigs_gaugeTheory_L40', 
		     'spinConfigs_gaugeTheory_L80']
	 
	data_files = ising + gauge + temperatures_Ising + temperatures_gauge
	
	data_types_config = ['spinConfigs_Ising','spinConfigs_gauge']
	data_types_temps = ['temperatures_Ising','temperatures_gauge']
	data_obj_format = {k: 'array' for k in data_types_config+data_types_temps}
	data_reps = ['pca','tsne']
	
	data_params =  {'data_files': data_files,
		            'data_types':data_types_config+data_types_temps,
					'data_format': 'npz', 
					'data_obj_format': data_obj_format,
					'data_dir': 'dataset/tsne/',
					'one_hot': [False]}
	 
	
	data,data_sizes,_,_ = Data_Process().importer(data_params,
							data_typing='dict',data_lists=True,upconvert=True)
	display(True,True,'Data Imported... \n'+str(data_sizes)+'\n')
	
	# Change keys structure for appropriate plot labels (i.e. L_XX size labels)
	ind_data = [-3,None]
	ind_type = [0,None]
		
	data_typed = dict_modify(data,data_types_config+data_types_temps,
				              f=lambda k,v: v.copy(),i=ind_data,j=ind_type)
	
	data_sizes = dict_modify(data_sizes,data_types_config+data_types_temps,
				              f=lambda k,v: v,i=ind_data,j=ind_type)
	
	data_keys = {t: sorted(list(d.keys())) for t,d in data_typed.items()}
	
	Y = {r: dict_modify(data,data_types_config,
				              f=lambda k,v: [],i=ind_data,j=ind_type)
			for r in data_reps}
	
	data_types_config = [t[slice(*ind_type)] for t in data_types_config]
	data_types_temps  = [t[slice(*ind_type)] for t in data_types_temps]
	
	
	# Setup Plotting
	plot_keys = {}
	plot_bool = {}
	for r in data_reps:
		for t in Y[r].keys():
			plot_keys[r+'_'+t] = data_keys[t]
			plot_bool[r+'_'+t]= True
	
	Data_Process().plot_close()        
	Data_Proc = Data_Process(plot_keys,plot_bool)
	
	comp = lambda x,i: {k:v[:,i] for k,v in x.items() if np.any(v)}
	
	
	print('keys',{t:d.keys() for t,d in data_typed.items()})
	print('keys',plot_keys)
	print('Y',Y)
	
	
	
	# tSNE and PCA Analysis

	for t in sorted(data_types_config):
		for k in data_keys[t]:       
			n = np.shape(data_typed[t][k])[0]
			N = data_sizes[t][k][1]
			M = 2
			tol = 1e-6
			perp = np.log(20.0)
			#print('Desired Perp: ',perp,n,data_typed[t][k])
			n_iter=50
			#d = pairwise_distance(data_typed[t][k])
			#x2p0(d,entropy_gaussian,tol,perp,n_iter)
			#binary_search(d,np.ones(n),entropy_gaussian,perp*np.ones(n),tol,n_iter)
			Y['pca'][t][k] = pca0(data_typed[t][k])
			Y['tsne'][t][k]   =  tsne0(data_typed[t][k],M,N,perp)
#            
#            for r in data_reps:
#                display(1,1,'%s for %s: %s...%s'%(r,t,k,
#                                              str(np.shape(Y[r][t][k]))))
#            
#            
#            
		for r in data_reps:		
			Data_Proc.plotter(comp(Y[r][t],1),comp(Y[r][t],0),
                             plot_props(Y[r][t].keys(),r,t[-5:]),
                             data_key=r+'_'+t)
			Data_Proc.exporter({r+'_'+t: Y[r][t]},data_params)
	Data_Proc.plot_save(data_params,read_write='a')
#            
#    
#    
#    
