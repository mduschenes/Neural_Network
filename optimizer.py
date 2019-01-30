# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt Duschenes
Machine Learning and Many Body Physics
PSI 2018 - Perimeter Institute
"""
import sys
sys.path.insert(0, "$[HOME]/google-drive-um/PSI/"+
                   "PSI Essay/PSI Essay Python Code/tSNE_Potts/")

import optimizer_methods
from data_functions import Data_Process 
from misc_functions import (display,array_sort,caps,line_break,
                              dict_reorder,index_nested,dict_feed)

import numpy as np
import tensorflow as tf

tf.reset_default_graph()

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)



class optimizer(object):
    
    def __init__(self,alg_params = {
                     'n_neuron': [None,100,None],                  
                     'neuron_func':{ 'layer':  tf.nn.sigmoid,
                                     'output': tf.nn.sigmoid
                                  },
                     'cost_functions': {
                         'cross_entropy': lambda x_,y_,y_est,eps= 10**(-8): 
                                         -tf.reduce_sum(
                                         y_*tf.log(y_est+eps) +
                                         (1.0-y_)*tf.log(1.0-y_est +eps),
                                         axis=1),
                                                 
                          'mse': lambda x_,y_,y_est: (1/2)*(
                                        tf.reduce_sum(tf.square(y_est-y_),
                                        axis=1)),
                          
                          'entropy_logits': lambda x_,y_,y_est: 
                                  tf.nn.sigmoid_cross_entropy_with_logits(
                                                 labels=y_, logits=y_est),
                          'L2': lambda var,eta_reg: tf.add_n([ tf.nn.l2_loss(v) 
                                         for v in var]) * eta_reg
                            },
                                      
                    'optimize_func': {'grad': lambda a,c: 
                                            tf.train.GradientDescentOptimizer(
                                                                a).minimize(c),
                                     'adam': lambda a,c: 
                                            tf.train.AdamOptimizer(
                                                                a).minimize(c)
                                },
                           
                  }):
                
        # Define Neural Network Properties Dictionary
        self.alg_params = alg_params  
       
        return


    def training(self,data_params = 
                     {'data_files': ['x_train','y_train','x_test','y_test'],
                      'data_sets': ['x_train','y_train','x_test','y_test'],
                      'data_types': ['train','test','other'],
                      'data_wrapper': lambda i,v: array_sort(i,v,0,'list'), 
                      'label_titles': ['High','Low'],
                      'data_format': 'npz',
                      'data_dir': 'dataset/',
                      'one_hot': False,
                      'upconvert': True,
                      'data_lists': True
                     },
                     plot_params={},
                     train=True,
                     timeit = True, printit=True,
                     save = True, plot = False,**kwargs):
    
        # Initialize Network Data and Parameters
        data_struct = 0
        display(True,True,'Neural Network Starting...')

        # Import Data
        Data_Process().plot_close()
        data_params['data_dir'] += self.alg_params['method']+'/'


        
        data,data_size,data_typed,data_keys=Data_Process().importer(
                                                  data_params)

                      
        
        display(True,True,'Data Imported...%s'%data_size)

        for t in data_params['data_types']:
            if not any(t in key for key in data.keys()):
                try:
                  setattr(locals(), t, False)
                except AttributeError:
                  kwargs[t] = False
        
        
        
        # Define Number of Neurons at Input and Output
        train_keys = list(data_typed[data_params['data_types'][0]].keys())
        test_keys = list(data_typed[data_params['data_types'][1]].keys())
        self.alg_params['n_dataset_train'],self.alg_params['n_neuron'][0] = (
                                        data_size[train_keys[0]])
        self.alg_params['n_neuron'][-1] = data_size[train_keys[1]][1]
        self.alg_params['n_dataset_test'] = data_size.get(test_keys[0],[None])[0]
        
        # Define Training Parameters
        self.alg_params['n_batch_train'] = max(1,
                                          int(self.alg_params['n_batch_train']*
                                            self.alg_params['n_dataset_train']))
        self.alg_params['n_epochs_meas'] = max(1,
                                          int(self.alg_params['n_epochs']*
                                             self.alg_params['n_epochs_meas']))
        
    
    
    
        # Initialize Optimization Method (i.e Neural Network Layers)
        y_est,x_,y_,T,W,b = getattr(optimizer_methods,
                              self.alg_params.get('method','neural_network'))(
                                              self.alg_params)

            
        
        display(True,True,self.alg_params.get('method','neural_network')+ 
                                                          ' Initialized...')
        
               
        # Initalize Tensorflow session
        sess = tf.Session()
        
        # Session Run Function
        sess_run = lambda var,data: sess.run(var,feed_dict={k:v 
                                        for k,v in zip([x_,y_],data.values())})
                  

        
        # Define Cost Function (with possible Regularization)
        cost0 = lambda x_,y_,y_est: tf.reduce_mean(
                                    self.alg_params['cost_functions'][
                                     self.alg_params['cost_func']](x_,y_,y_est))
        
        if self.alg_params['regularize']:
            cost_reg = lambda x_,y_,y_est: (
                                        self.alg_params['cost_functions'][
                                        self.alg_params['regularize']](
                                          var=W,
                                          eta_reg=self.alg_params['eta_reg']))
                                        #v for v in tf.trainable_variables() 
                                                       # if 'reg' in v.name],
        else:
            cost_reg = lambda x_,y_,y_est: 0
        
        cost = lambda x_,y_,y_est: cost0(x_,y_,y_est) + cost_reg(x_,y_,y_est)
                                          
                                           
        
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = self.alg_params['alpha_learn']
#        tf.train.exponential_decay(self.alg_params['apha_learn'],
#                              global_step, self.alg_params['n_epochs'],
#                              0.96, staircase=True)
        

        # Training Output with Learning Rate Alpha and regularization Eta
        train_step = self.alg_params['optimize_functions'][
                                 self.alg_params['optimize_func']](
                                                      cost=cost(x_,y_,y_est),
                                                      alpha_learn=alpha_learn)
        if 'cost' in process_params.keys():
          process_params['cost']['function'] = cost
        else:
          process_params['cost'] = { 
                                'domain':['other','epochs'],'data': [],
                                'data_name': 'Cost',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': cost}

        
        display(True,True,'Results Initialized...')
           
        
        
        
        # Initialize Plotting
        plot_keys = dict_reorder(dict_reorder(process_params,'plot_type',True))
        plot_loop = plot.pop('plot_loop',True)
        Data_Proc = Data_Process(plot=plot, keys = plot_keys)

        if plot_params == {}:
          plot_params = {
                     k: {
                    
                      'ax':   {'title' : '',
                                'xlabel': caps(process_params[k]['domain_name'],
                                                True,split_char='_'), 
                                'ylabel': caps(process_params[k]['data_name'],
                                                True,split_char='_'), 
                                },
                      
                      'plot':  {'marker':'*' ,
                                'color':np.random.rand(3) if 
                                                k != 'y_est' else None},
                      
                      'data':  {'plot_type':'plot'},
                                
                      'other': {'label': lambda x='':x,
                                'plot_legend':True if k == 'y_est' else False,
                                'sup_legend': False,
                                'sup_title': {'t': 'Optimization Parameters:'\
                                               '\n \n'+ 
                                               '\n'.join([str(caps(k,True,
                                                              split_char='_'))+
                                                ':  '+line_break(str(
                                                  self.alg_params[k]),15,
                                                  line_space=' '*int(
                                                            2.5*len(str(k))))
                                                    for i,k in enumerate(
                                                    sorted(list(
                                                    self.alg_params.keys())))]),
                                              'x':0.925,'y':0.97,'ha':'left',
                                              'fontsize':7},
                                  'pause':0.01
                                }
                     }
                    for k in process_params.keys()}

        
        
        
        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        display(True,True,'Training...')



        # Train Model
        if vars().get('train',kwargs.get('train')):
            epoch_range = range(self.alg_params['n_epochs'])
            dataset_range = np.arange(self.alg_params['n_dataset_train'])
            batch_range = range(0,self.alg_params['n_dataset_train'],
                                    self.alg_params['n_batch_train'])

            
            # # Check if Training and Testing Data Exists
            # data_typed_keys = ['train','test']
            # for t in data_typed.keys():
            #   if all([i in t for i in data_typed_keys]):
            #     break
            #   if data_typed_keys[0] in t and not data_typed_keys[1] in t:
            #     data_typed
            # if not data_typed.get('train'):
            #   if not data_typed.get('test')


            # Train Model over n_epochs with Gradient Descent 
            for epoch in epoch_range:


                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                np.random.shuffle(dataset_range)                       
                for i_batch in batch_range:
                    
                    # Choose Random Batch of data                    
                    sess_run(train_step,{k:d[dataset_range,:][i_batch:
                                   i_batch + self.alg_params['n_batch_train'],:]
                                   for k,d in data_typed[
                                   data_params['data_types'][0]].items()})
                
            
                # Record Results every n_epochs_meas
                if (epoch+1) % self.alg_params['n_epochs_meas'] == 0:

                    display(printit,False,'\nEpoch: %d'% epoch)

                    data_typed['other']['epochs'] = epoch+1
              
                    for key,process in process_params.items():
                        if (vars().get(process['data_type']) is not False) and (
                            kwargs.get(process['data_type']) is not False) :
                          y,x = process.get('data_wrapper',
                              lambda y,x:(y,x))(
                              sess_run(process['function'](x_,y_,y_est),
                              index_nested(data_typed,process['data_type'])),
                              index_nested(data_typed,process['domain_type']))
                          x = np.reshape(x,(-1))

                          if y.dtype=='object':
                            if all([((np.ndim(yi)>1) and 
                                    (1 not in np.shape(yi)[1:])) for yi in y]):
                              y = {
                                    t: np.array([np.reshape(
                                                np.mean(np.take(yi,j,-1),-1),
                                                (-1)) for yi in y])
                                    for j,t in enumerate(process.get('labels',
                                                    [process['data_name']]))}
                              x = {t: x for t in process.get('labels',
                                                    [process['data_name']])}
                            else:
                              y = np.reshape(np.array(
                                          [np.mean(yi,-1) for yi in y]),(-1))

                          elif np.ndim(y) > 2:
                            y = {
                                  t: np.reshape(np.mean(np.take(y,j,-1),
                                                                -1),(-1))
                                  for j,t in enumerate(process.get('labels',
                                                  [process['data_name']]))}
                            x = {t: x for t in process.get('labels',
                                                    [process['data_name']])}
                          elif np.ndim(y) == 2:
                            y = np.reshape(np.mean(y,-1),(-1))

                          if ((np.size(x) > 1 ) or (np.size(y) > 1 )) or (
                            isinstance(x,dict) and isinstance(y,dict)):
                            process['data'] = y
                            process['domain'] = x
                          else:  
                            process['data'].append(y)
                            process['domain'].append(x)

                          if process['plot_type'] in ['Training and Testing']:
                            display(printit,False,'%s: %s'%(
                                                  process['data_name'],str(y)))

                    display(printit,True,'Time: ')


                    # Save and Plot Data
                    if plot_loop:
                      Data_Proc.plotter(data=dict_reorder(process_params,'data',True),
                              domain=dict_reorder(process_params,'domain',True),
                              plot_props=plot_params)        
     
                
                    
        
        # Plot and Save Final Results
        display(True,False,'Saving Data...')
        plot_label = '%d epochs'%self.alg_params['n_epochs']
        if not plot_loop:
          Data_Proc.plot_set(plot=True,keys=plot_keys)
          Data_Proc.plotter(data=dict_reorder(process_params,'data',True),
                              domain=dict_reorder(process_params,'domain',True),
                              plot_props=plot_params)
                            
        Data_Proc.plot_save(data_params,label=plot_label)

        # Pause before exiting
        input('Press [Enter] to exit')


    
    def testing(self,results,data,data_func):
        
        for key,val in results.items():
            val.append(data_func.get(key,lambda x: x)(data))
            
        return results

    






if __name__ == '__main__':
    

    # Dataset Parameters
    data_params =  {'data_files': ['x_test','y_test',
                                    'T_other'],
                    # 'data_sets': ['x_train','y_train','x_test','y_test',
                    #                 'T_other'],
                    'data_types': ['train','test','other'],
                    'data_format': 'npz',
                    'data_dir': 'dataset/',
                    'one_hot': [True,'y_'],
                    'upconvert': True,
                    'data_lists': True,
                    'data_seed':{ 'train':{'seed_type':'test',
                                  'seed_dimensions':[0,np.random.choice(9999,8000,replace=False)],
                                  'remove_seed':True}}
                   }

    process_params = {'acc_train':{'domain':[],'data': [],
                                'data_name': 'Training Accuracy',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                             tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))},
                  'acc_test': { 'domain':[],'data': [],
                                'data_name': 'Testing Accuracy',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                            tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))},
                  'cost': { 'domain':[],'data': [],
                                'data_name': 'Cost',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est,eps= 10**(-8): 
                                               -tf.reduce_sum(
                                               y_*tf.log(y_est+eps) +
                                               (1.0-y_)*tf.log(1.0-y_est +eps),
                                               axis=1)},
                  'y_equiv':  { 'domain':[],'data': [],
                                'data_name': 'Average Output',
                                'domain_name': 'Temperature',
                                'data_type': 'test',
                                'domain_type': ['other','T_other'],
                                'plot_type': 'Model Predictions',
                                'labels':[],
                                'data_wrapper': lambda v,i: array_sort(v,i,0,
                                                        'ndarray'),
                                'function': lambda x_,y_,y_est: tf.equal(
                                                 tf.argmax(y_est,axis=1),
                                                 tf.argmax(y_,axis=1))},
                  'y_est':    { 'domain':[],'data': [],
                                'data_name': 'Phase Label',
                                'domain_name': 'Temperature',
                                'data_type': 'test',
                                'domain_type': ['other','T_other'],
                                'plot_type': 'Model Predictions',
                                'labels':['T<Tc','T>Tc'],
                                'data_wrapper': lambda v,i: array_sort(v,i,0,
                                                            'ndarray'),
                                'function': lambda x_,y_,y_est: y_est},
                  'y_grad':   { 'domain':[],'data': [],
                                'data_name': 'Output Gradient',
                                'domain_name': 'Temperature',
                                'data_type': 'test',
                                'domain_type': ['other','T_other'],
                                'plot_type': 'Gradients',
                                'labels':[],
                                'data_wrapper': lambda v,i: array_sort(v,i,0,
                                                        'ndarray'),
                                'function': lambda x_,y_,y_est:
                                        tf.math.reduce_sum(tf.gradients(
                                              y_est,x_),axis=[0,-1])}
                  }



    kwargs = {'train':True, 'test':True, 'other':True,
              'plot':{'plot_loop':False,'Training and Testing':True,
                      'Model Predictions':True,'Gradients':True},
              'save':True, 'printit':False, 'timeit':False}


    # Neural Network Parameters    
    network_params = {
                  'n_neuron': [None,100,None], 
                  'alpha_learn': 0.0035, 'eta_reg': 0.0005,'sigma_var':0.1,                  
                  'n_epochs': 5,'n_batch_train': 1/10,'n_epochs_meas': 1/50,
                  'cost_func': 'cross_entropy', 'optimize_func':'adam',
                  'regularize':'L2',
                  'method': 'neural_network',
                  'layers':'fcc' ,                
                  'neuron_func':{
                   'layer':  tf.nn.sigmoid,
                   'output': tf.nn.sigmoid
                              },
                   'cost_functions': {
                     'cross_entropy': lambda x_,y_,y_est,eps= 10**(-8): 
                                     -tf.reduce_sum(
                                     y_*tf.log(y_est+eps) +
                                     (1.0-y_)*tf.log(1.0-y_est +eps),
                                     axis=1),
                                             
                      'mse': lambda x_,y_,y_est: (1/2)*(
                                    tf.reduce_sum(tf.square(y_est-y_),
                                    axis=1)),
                      
                      'entropy_logits': lambda x_,y_,y_est: 
                              tf.nn.sigmoid_cross_entropy_with_logits(
                                             labels=y_, logits=y_est),
                      'L2': lambda **kwargs: tf.add_n([ tf.nn.l2_loss(v) 
                                    for v in kwargs['var']]) * kwargs['eta_reg']
                            },
                                  
                  'optimize_functions': {'grad': lambda cost,**kwargs: 
                                        tf.train.GradientDescentOptimizer(
                                                    alpha_learn).minimize(cost),
                                 'adam': lambda cost,**kwargs: 
                                        tf.train.AdamOptimizer(
                                         kwargs['alpha_learn']).minimize(cost)
                            },
                           
                  }
  
    # Plot Parameters
    plot_params = {
                     key: {
                    
                      'ax':   {'title' : '',
                               'xlabel':caps(process_params[key]['domain_name'],
                                                True,split_char='_'), 
                                'ylabel': caps(process_params[key]['data_name'],
                                                True,split_char='_'), 
                                },
                      
                      'plot':  {'marker':'*' ,
                                'color':np.random.rand(3) if (
                                          key != 'y_est') else None},
                      
                      'data':  {'plot_type':'plot'},
                                
                      'other': {'label': lambda x='':x,
                                'plot_legend':True if key == 'y_est' else False,
                                'sup_legend': False,
                                'sup_title': {'t': 'Optimization Parameters:'\
                                               '\n \n'+ 
                                               '\n'.join([str(caps(k,True,
                                                              split_char='_'))+
                                                ':  '+line_break(str(v),15,
                                                  line_space=' '*int(
                                                            2.5*len(str(k))))
                                                     for i,(k,v) in enumerate(
                                                     sorted(list(
                                                    network_params.items())))]),
                                              'x':0.925,'y':0.97,'ha':'left',
                                              'fontsize':7},
                                  'pause':0.01
                                }
                     }
                    for key in process_params.keys()}
    
    
    
    # Run Neural Network
    data_struct = 0   
    nn = optimizer(network_params)
    nn.training(data_params,plot_params,**kwargs)
    
    