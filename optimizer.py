# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 11:04:42 2018

@author: Matt Duschenes
Machine Learning and Many Body Physics
PSI 2018 - Perimeter Institute
"""
import sys
sys.path.insert(0, "C:/Users/Matt/Google Drive/PSI/"+
                   "PSI Essay/PSI Essay Python Code/tSNE_Potts/")

import optimizer_methods
from data_functions import Data_Process 
from misc_functions import display,array_sort,caps,line_break

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
                     'cost_func': {
                          'cross_entropy': lambda y_label,y_est,eps= 10**(-8): 
                                         -tf.reduce_sum(
                                         y_label*tf.log(y_est+eps) +
                                         (1.0-y_label)*tf.log(1.0-y_est +eps),
                                         axis=1),
                                                 
                          'mse': lambda y_label,y_est: (1/2)*(
                                        tf.reduce_sum(tf.square(y_est-y_label),
                                        axis=1)),
                          
                          'entropy_logits': lambda y_label,y_est: 
                                  tf.nn.sigmoid_cross_entropy_with_logits(
                                                 labels=y_label, logits=y_est),
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
                     train = True, test = False, other=False,
                     timeit = True, printit=True,
                     save = True, plot = False):
    
        # Initialize Network Data and Parameters
        
        display(True,True,'Neural Network Starting...')

        # Import Data
        Data_Process().plot_close()
        data_params['data_dir'] += self.alg_params['method']+'/'


        
        data,data_size,data_typed,data_keys=Data_Process().importer(
                                                  data_params)

                      
        
        display(True,True,'Data Imported...%s'%data_size)

                
        for t in data_params['data_types']:
            if not any(t in key for key in data.keys()):
                setattr(locals(), t, False)
        
        
        
        # Define Number of Neurons at Input and Output
        self.alg_params['n_dataset_train'],self.alg_params['n_neuron'][0] = (
                                                          data_size['x_train'])
        self.alg_params['n_neuron'][-1] = data_size['y_train'][1]
        self.alg_params['n_dataset_test'] = data_size.get('x_test',[None])[0]
        
        # Define Training Parameters
        self.alg_params['n_batch_train'] = max(1,int(self.alg_params['n_batch_train']*
                                                self.alg_params['n_dataset_train']))
        self.alg_params['n_epochs_meas'] = max(1,int(self.alg_params['n_epochs']*
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
                       
        # Initialize Lable, Accuracy and Cost Functions
        data_struct = 1
        if data_struct:
          y_equiv = tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32)
          train_acc =  tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))
          test_acc =  tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))

        


        
       



        
        # Define Cost Function (with possible Regularization)
        cost = tf.reduce_mean(self.alg_params['cost_functions'][
                                            self.alg_params['cost_func']](y_,y_est))
        
        if self.alg_params['regularize']:
            cost += self.alg_params['cost_functions'][
                 self.alg_params['regularize']](W,self.alg_params['eta_reg'])
                                          #v for v in tf.trainable_variables() 
                                                       # if 'reg' in v.name],
                                           
        
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = self.alg_params['alpha_learn']
#        tf.train.exponential_decay(self.alg_params['apha_learn'],
#                              global_step, self.alg_params['n_epochs'],
#                              0.96, staircase=True)
        

        # Training Output with Learning Rate Alpha and regularization Eta
        train_step = self.alg_params['optimize_functions'][
                                 self.alg_params['optimize_func']](
                                                      cost=cost,
                                                      alpha_learn=alpha_learn)

         
       
        
        # Initialize Results Dictionaries
        if data_struct:
          results_keys = {}
          results_keys['train'] = ['train_acc','cost']
          results_keys['test'] =  ['test_acc']
          results_keys['other'] = ['y_equiv','y_est']
          results_keys['all'] = (results_keys['train']+results_keys['test']+
                              results_keys['other'])
               
          loc = vars()
          loc = {key:loc.get(key) for key in results_keys['all'] 
                                 if loc.get(key) is not None}
          results_keys['all'] = loc.keys()        
          
          
          # Results Dictionary of Array for results values
          results = {key: [] for key in results_keys['all']}
                             
          # Results Dictionary of Functions for results       
          results_func = {}
          for key in results_keys['all']:
              results_func[key] = lambda feed_dict : sess_run(loc[key],feed_dict) 
          
        
        display(True,True,'Results Initialized...')
           
        
        
        
        
        # Initialize Plotting
        plot_keys = {}
        if test and train:
          plot_keys['Testing and Training'] = (results_keys['train']+ 
                                              results_keys['test'])
        elif test and not train:
          plot_keys['Testing'] = results_keys['test']
        elif not test and train:
          plot_keys['Testing'] = results_keys['test']

        if other:
          plot_keys['Other'] = results_keys['other']
        
        Data_Proc = Data_Process(plot=plot, keys = plot_keys)
        
        if plot_params == {}:
            plot_params = {
                     k: {
                    
                      'ax':   {'title' : '',
                                'xlabel': 'Epochs' if k 
                                           not in results_keys['other'] 
                                           else 
                                           caps(str(data_keys['other'][0]
                                                ).replace('_other','')), 
                                'ylabel': caps(k,True,split_char='_')
                                },
                      
                      'plot':  {'marker':'*','color':np.random.rand(3)},
                      
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
                    for k in results_keys['all']}

        
        
        
        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        display(True,True,'Training...')
        


        # Train Model
        if train:

            epoch_range = range(self.alg_params['n_epochs'])

            # data_typed['other']['epochs'] = epoch_range


            dataset_range = np.arange(self.alg_params['n_dataset_train'])
            batch_range = range(0,self.alg_params['n_dataset_train'],
                                    self.alg_params['n_batch_train'])

            def domain(i = self.alg_params['n_epochs'],
                       f = lambda i,*k: list(range(*i))):
                
                if not hasattr(i,'__iter__'):
                    i = (0,i,self.alg_params['n_epochs_meas'])
                
                if callable(f):
                    return {k: f(i,k) 
                            for k in results_keys['all'] }
                else:
                    return {k: f for k in results_keys['all']}
            
                        
            # Train Model over n_epochs with Gradient Descent 
            for epoch in epoch_range:
                           
                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                np.random.shuffle(dataset_range)                       
                for i_batch in batch_range:
                    
                    # Choose Random Batch of data                    
                    sess_run(train_step,{k:d[dataset_range,:][i_batch:
                                   i_batch + self.alg_params['n_batch_train'],:]
                                   for k,d in data_typed['train'].items()})
                
            
                # Record Results every n_epochs_meas
                if (epoch+1) % self.alg_params['n_epochs_meas'] == 0:
                    
                    # Record Results: Cost and Training Accuracy
                    for key,val in results.items():
                        if train and key in results_keys['train']:
                            val.append(results_func[key](data_typed['train'])) 
                        
                        elif test and key in results_keys['test']:
                            val.append(results_func[key](data_typed['test']))
                        
                        elif other and key in results_keys['other']:
                            val.append(results_func[key](data_typed['test']))
                                                


                     # Display Results
                    display(printit,timeit,'\nEpoch: %d'% epoch + 
                          '\n'+
                          'Training Accuracy: '+str(results['train_acc'][-1])+
                          '\n'+
                          'Testing Accuracy: '+str(results['test_acc'][-1])+
                          '\n'+
                          'Cost:             '+str(results['cost'][-1])+
                          '\n')

            
                    # Save and Plot Data
                    #list(range(*i))) #domain(epoch+1),
                    Data_Proc.plotter(results,list(range(0,epoch+1,
											      self.alg_params['n_epochs_meas'])),
                            plot_props=plot_params,
                            data_key='Testing and Training')
            
            

    
        # Make Results Class variable for later testing with final network
        data_params['results'] = results
        data_params['results_func'] = results_func

        self.data_params = data_params
        
       
        # Process Other Data with final trained values
        if other: 
            # Sort and Average over each array from other results
            results_sort_avg = [{k:{} for k in results_keys['other']}
                                  for i in range(len(data_typed['other']))]
            domain_sort_avg = [{k:[] for k in results_keys['other']}
                                  for i in range(len(data_typed['other']))]
                        
            for i,(k_other,d_other) in enumerate(data_typed['other'].items()):

                for k in results_keys['other']:
                    results_sort,d_sort = data_params.get('data_wrapper',lambda x,y: x)(results[k][-1],d_other)
                    if np.ndim(results_sort) > 2 and (
                      np.shape(results_sort)[-1] == 
                      np.size(data_params.get('label_titles',[]))):

                        results_sort_avg[i][k] = {
                            t: np.reshape(np.mean(np.take(results_sort,j,-1),-1),(-1))
                            for j,t in enumerate(data_params['label_titles'])}

                        domain_sort_avg[i][k] = {t: d_sort for t in 
                                                  data_params['label_titles']}

                        results[k] = results_sort_avg[i][k]

                        
                    else:
                        results[k] = np.reshape(np.mean(
                                                        results_sort,-1),(-1))
                        domain_sort_avg[i][k] = d_sort
                if plot:
                  Data_Proc.plotter(results,
                                   domain_sort_avg[i],plot_params, 'Other')
                
                
                    
        
        # Plot and Save Final Results
        display(True,False,'Saving Data...')
        plot_label = '%d epochs'%self.alg_params['n_epochs']
        if not plot:
            Data_Proc_Final = Data_Process(plot=True,
                                           keys = {'Testing and Training':
                                                         results_keys['train']+
                                                         results_keys['test'],
                                         'Other': results_keys['other']})
            Data_Proc_Final.plotter(results,list(range(0,epoch+1,
									self.alg_params['n_epochs_meas'])),
									plot_params, 'Testing and Training')
            if other:
                for i in range(len(data_typed['other'])):
                  Data_Proc_Final.plotter(results,
                                   domain_sort_avg[i],plot_params, 'Other')
            
            Data_Proc_Final.plot_save(data_params,label=plot_label)
        
        else:
            Data_Proc.plot_save(data_params,label=plot_label)

        # Pause before exiting
        input('Press [Enter] to exit')


    
    def testing(self,results,data,data_func):
        
        for key,val in results.items():
            val.append(data_func.get(key,lambda x: x)(data))
            
        return results

    






if __name__ == '__main__':
    

    # Dataset Parameters
    data_params =  {'data_files': ['x_train','y_train','x_test','y_test',
                                    'T_other'],
                    # 'data_sets': data_sets,
                    'data_types': ['train','test','other'],
                    'data_wrapper': lambda v,i: array_sort(v,i,0,'ndarray'),
                    'data_format': 'npz',
                    'label_titles':['T>Tc','T<Tc'],
                    'data_dir': 'dataset/',
                    'one_hot': [True,'y_'],
                    'upconvert': True,
                    'data_lists': True
                   }

    process_params = {'acc_train':{ 'domain':['other','epochs'],'data': [],
                                'data_type': 'train',
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                             tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))},
                  'acc_test': { 'domain':['other','T_other'],'data': [],
                                'data_type': 'test',
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                            tf.reduce_mean(tf.cast(tf.equal(
                                                      tf.argmax(y_est,axis=1),
                                                      tf.argmax(y_,axis=1)), 
                                                      tf.float32))},
                  'y_equiv':  { 'domain':['other','T_other'],'data': [],
                                'data_type': 'train',
                                'plot_type': 'Model Predictions',
                                'labels':[],
                                'function': lambda x_,y_,y_est: tf.equal(
                                                 tf.argmax(y_est,axis=1),
                                                 tf.argmax(y_,axis=1))},
                  'y_est':    { 'domain':['other','T_other'],'data': [],
                                'data_type': 'train',
                                'plot_type': 'Model Predictions',
                                'labels':['T>Tc','T<Tc'],
                                'function': lambda x_,y_,y_est: y_est},
                  'y_grad':   { 'domain':['train','x_train'],'data': [],
                                'data_type': 'train',
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est:
                                                      tf.gradients(y_est,x_)}
                  }



    kwargs = {'train':True,'test':True,'other':True,
              'plot':True,'save':True,'printit':True,'timeit':True}


    # plot_params = {
    #      k: {
        
    #       'ax':   {'title' : '',
    #                 'xlabel': 'Epochs' if k 
    #                            not in results_keys['other'] 
    #                            else 
    #                            caps(str(data_keys['other'][0]
    #                                 ).replace('_other','')), 
    #                 'ylabel': caps(k,True,split_char='_')
    #                 },
          
    #       'plot':  {'marker':'*','color':np.random.rand(3)},
          
    #       'data':  {'plot_type':'plot'},
                    
    #       'other': {'label': lambda x='':x,
    #                 'plot_legend':True if k == 'y_est' else False,
    #                 'sup_legend': False,
    #                 'sup_title': {'t': 'Optimization Parameters:'\
    #                                '\n \n'+ 
    #                                '\n'.join([str(caps(k,True,
    #                                               split_char='_'))+
    #                                 ':  '+line_break(str(
    #                                   self.alg_params[k]),15,
    #                                   line_space=' '*int(
    #                                             2.5*len(str(k))))
    #                                      for i,k in enumerate(
    #                                      sorted(list(
    #                                      self.alg_params.keys())))]),
    #                               'x':0.925,'y':0.97,'ha':'left',
    #                               'fontsize':7},
    #                   'pause':0.01
    #                 }
    #      }
    #     for k in process_params.keys()}

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
                     'cross_entropy': lambda y_label,y_est,eps= 10**(-8): 
                                     -tf.reduce_sum(
                                     y_label*tf.log(y_est+eps) +
                                     (1.0-y_label)*tf.log(1.0-y_est +eps),
                                     axis=1),
                                             
                      'mse': lambda y_label,y_est: (1/2)*(
                                    tf.reduce_sum(tf.square(y_est-y_label),
                                    axis=1)),
                      
                      'entropy_logits': lambda y_label,y_est: 
                              tf.nn.sigmoid_cross_entropy_with_logits(
                                             labels=y_label, logits=y_est),
                      'L2': lambda var,eta_reg: tf.add_n([ tf.nn.l2_loss(v) 
                                         for v in var]) * eta_reg
                            },
                                  
                  'optimize_functions': {'grad': lambda cost,alpha_learn: 
                                        tf.train.GradientDescentOptimizer(
                                                    alpha_learn).minimize(cost),
                                 'adam': lambda cost,alpha_learn: 
                                        tf.train.AdamOptimizer(
                                                    alpha_learn).minimize(cost)
                            },
                           
                  }
  
    
    
    
    # Run Neural Network   
    nn = optimizer(network_params)
    nn.training(data_params,**kwargs)
    
    