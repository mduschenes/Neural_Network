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
from misc_functions import (DISPLAY,array_sort,caps,line_break,
                              dict_reorder,index_nested,dict_feed)

import numpy as np
import tensorflow as tf
import datetime

tf.reset_default_graph()

seed=1234
np.random.seed(seed)
tf.set_random_seed(seed)


network_functions = {
  'cost_functions': {
                 'cross_entropy': lambda x_,y_,y_est,eps= 10**(-8): 
                                 -tf.reduce_sum(
                                 y_*tf.log(y_est+eps) +
                                 (1.0-y_)*tf.log(1.0-y_est +eps),
                                 axis=1),
                                         
                  'mse': lambda x_,y_,y_est: (1/2)*(
                                tf.reduce_mean(tf.square(y_est-y_),
                                axis=1)),
                  'mse_tf': lambda x_,y_,y_est: 
                                tf.losses.mean_squared_error(y_,y_est),                  
                  'entropy_logits': lambda x_,y_,y_est: 
                          tf.nn.sigmoid_cross_entropy_with_logits(
                                         labels=y_, logits=y_est)
                    },
  'regularize_functions':{
                  'L2': lambda **kwargs: tf.add_n([ tf.nn.l2_loss(v) 
                                for v in kwargs['var']]) * kwargs['eta_reg']
                        },      
  'optimize_functions': {
                  'grad': lambda cost,**kwargs: 
                        tf.train.GradientDescentOptimizer(
                                    kwargs['alpha_learn']).minimize(cost),
                 'adam': lambda cost,**kwargs: 
                        tf.train.AdamOptimizer(
                         kwargs['alpha_learn']).minimize(cost)
            },
  'activation_functions': {
                  'sigmoid': tf.nn.sigmoid}
}


class optimizer(object):
    
    def __init__(self):     
        return


    def training(self,alg_params,data_params,process_params,
                     plot_params={},
                     train=True,
                     timeit = True, printit=True,
                     save = True, plot = False,**kwargs):
    
        # Initialize Network Data and Parameters
        display = DISPLAY().display
        display(True,False,'Neural Network Starting...'+
                            str(datetime.datetime.now()))

        # Import Data
        Data_Process().plot_close()
        data_params['data_dir'] += alg_params['method']+'/'


        
        data,data_size,data_typed,data_keys=Data_Process().importer(
                                                  data_params)

                      
        
        display(True,False,'Data Imported...%s'%data_size)

        for t in data_params['data_types']:
            if not any(t in key for key in data.keys()):
                try:
                  setattr(locals(), t, False)
                except AttributeError:
                  kwargs[t] = False
        
        
        
        # Define Number of Neurons at Input and Output
        train_keys = list(data_typed[data_params['data_types'][0]].keys())
        test_keys = list(data_typed[data_params['data_types'][1]].keys())
        alg_params['n_dataset_train'],alg_params['n_neuron'][0] = (
                                        data_size[train_keys[0]])
        alg_params['n_neuron'][-1] = data_size[train_keys[1]][1]
        alg_params['n_dataset_test']= data_size.get(test_keys[0],[None])[0]
        
        # Define Training Parameters
        alg_params['n_batch_train'] = max(1,
                                          int(alg_params['n_batch_train']*
                                            alg_params['n_dataset_train']))
        alg_params['n_epochs_meas'] = max(1,
                                          int(alg_params['n_epochs']*
                                             alg_params['n_epochs_meas']))
        
    
    
    
        # Initialize Optimization Method (i.e Neural Network Layers)
        try:
          y_est,x_,y_,T,W,b = getattr(optimizer_methods,
                              alg_params.get('method','neural_network'))(
                                              alg_params)
        except AttributeError:
          y_est,x_,y_,T,W,b = getattr(optimizer_methods,'neural_network')(
                                              alg_params)

            
        
        # display(True,False,alg_params.get('method','neural_network')+ 
        #                                   ' Initialized...')
        
        
               
        # Initalize Tensorflow session
        sess = tf.Session()
        
        # Session Run Function
        sess_run = lambda var,data: sess.run(var,feed_dict={k:v 
                                        for k,v in zip([x_,y_],data.values())})
                  

        
        # Define Cost Function (with possible Regularization)
        cost0 = lambda x_,y_,y_est: tf.reduce_mean(
                                    network_functions['cost_functions'][
                                     alg_params['cost_func']](x_,y_,y_est))
        
        if alg_params.get('regularize_func'):
            cost_reg = lambda x_,y_,y_est: (
                                network_functions['regularize_functions'][
                                alg_params['regularize_func']](
                                  var=W,
                                  eta_reg=alg_params['eta_reg']))
                                #v for v in tf.trainable_variables() 
                                               # if 'reg' in v.name],
        else:
            cost_reg = lambda x_,y_,y_est: 0
        
        cost = lambda x_,y_,y_est: cost0(x_,y_,y_est) + cost_reg(x_,y_,y_est)
                                          
                                           
        
        # Define Learning Rate Corrections
        #global_step = tf.Variable(0, trainable=False)
        alpha_learn = alg_params['alpha_learn']
#        tf.train.exponential_decay(alg_params['apha_learn'],
#                              global_step, alg_params['n_epochs'],
#                              0.96, staircase=True)
        

        # Training Output with Learning Rate Alpha and regularization Eta
        train_step = network_functions['optimize_functions'][
                                 alg_params['optimize_func']](
                                                      cost=cost(x_,y_,y_est),
                                                      alpha_learn=alpha_learn)
        if 'cost' in process_params.keys() and (
          process_params['cost'].get('function') is None):
          process_params['cost']['function'] = cost

        else:
          process_params['cost'] = { 'domain':[],'data': [],
                                'data_name': 'Cost',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': None}
           
        
        
        
        # Initialize Plotting
        plot_keys = dict_reorder(dict_reorder(process_params,'plot_type',True))
        plot_loop = plot.pop('plot_loop',True)
        kwargs['plot'] = plot
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
                                                  alg_params[k]),15,
                                                  line_space=' '*int(
                                                            2.5*len(str(k))))
                                                    for i,k in enumerate(
                                                    sorted(list(
                                                    alg_params.keys())))]),
                                              'x':0.925,'y':0.97,'ha':'left',
                                              'fontsize':7},
                                  'pause':0.01
                                }
                     }
                    for k in process_params.keys()}

        
        
        
        
        # Initialize all tensorflow variables
        sess.run(tf.global_variables_initializer())
        
        display(True,False,'Training...')

        # Train Model
        if vars().get('train',kwargs.get('train')):

            epoch_range = range(alg_params['n_epochs'])
            dataset_range = np.arange(alg_params['n_dataset_train'])
            batch_range = range(0,alg_params['n_dataset_train'],
                                    alg_params['n_batch_train'])
            if not data_typed.get('other'):
              data_typed['other'] = {}
            kwargs['train'] = train

            # Train Model over n_epochs with Gradient Descent 
            for epoch in epoch_range:


                # Divide training data into batches for training 
                # with Stochastic Gradient Descent
                np.random.shuffle(dataset_range)                       
                for i_batch in batch_range:
                    
                    # Choose Random Batch of data                   
                    sess_run(train_step,{k:d[dataset_range,:][i_batch:
                                   i_batch + alg_params['n_batch_train'],:]
                                   for k,d in data_typed[
                                   data_params['data_types'][0]].items()})
                
            
                # Record Results every n_epochs_meas
                if (epoch+1) % alg_params['n_epochs_meas'] == 0:

                    display(printit,False,'\nEpoch: %d'%(epoch+1))

                    data_typed['other']['epochs'] = epoch+1
              
                    for key,process in process_params.items():
                        if (vars().get(process['data_type']) is not False) and (
                            kwargs.get(process['data_type']) is not False) :
                          try:
                            y,x = process.get('data_wrapper',
                              lambda y,x:(y,x))(
                              sess_run(process['function'](x_,y_,y_est),
                              index_nested(data_typed,process['data_type'])),
                              index_nested(data_typed,process['domain_type']))
                          except TypeError:
                            y,x = process.get('data_wrapper',
                              lambda y,x:(None,None))(                              
                              index_nested(data_typed,process['data_type']), 
                                index_nested(data_typed,process['domain_type']))
                          

                          x = np.reshape(x,(-1))

                          if not callable(process.get('labels')):
                            labels = process.get('labels',
                                                    [process['data_name']])
                          else:
                            labels = process.get('labels')(x_,y_,y_est)

                          if y.dtype=='object':
                            if all([((np.ndim(yi)>1) and
                                    (1 not in np.shape(yi)[1:])) for yi in y]):
                              y = {
                                    t: np.array([np.reshape(
                                                np.mean(np.take(yi,j,-1),-1),
                                                (-1)) for yi in y])
                                    for j,t in enumerate(labels)}
                              x = {t: x for t in labels}
                            elif all([((np.ndim(yi)==1))for yi in y]) and (
                                  len(y) == len(labels)):
                              y = {
                                    t: y[j]
                                    for j,t in enumerate(labels)}
                              x = {t: x for t in labels}
                            else:
                              y = np.reshape(np.array(
                                          [np.mean(yi,-1) for yi in y]),(-1))

                          elif np.ndim(y) > 2:
                            y = {
                                  t: np.reshape(np.mean(np.take(y,j,-1),
                                                                -1),(-1))
                                  for j,t in enumerate(labels)}
                            x = {t: x for t in labels}

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
                      Data_Proc.plotter(
                              data=dict_reorder(process_params,'data',True),
                              domain=dict_reorder(process_params,'domain',True),
                              plot_props=plot_params)        
     
                
                    
        
        # Plot and Save Final Results
        display(True,True,'Training Complete\nPlotting and Saving Data...',t0=3)
        plot_label = '%d epochs'%alg_params['n_epochs']
        if not plot_loop:
          Data_Proc.plot_set(plot=True,keys=plot_keys)
          Data_Proc.plotter(data=dict_reorder(process_params,'data',True),
                              domain=dict_reorder(process_params,'domain',True),
                              plot_props=plot_params)
                            
        # Save Final Plots
        Data_Proc.plot_save(data_params,label=plot_label,read_write='w')

        # Save Final Results
        params_keys = ['data_params','alg_params','plot_params',
                        'process_params','kwargs']
        # for p in params_keys:
        #   print(p,locals().get(p))
        loc_vars = locals()
        params = {'params': {p: loc_vars.get(p) for p in params_keys}}
        # Data_Proc.exporter(params,data_params,format='npy')

        # Pause before exiting
        display(True,False,'Neural Network Complete... '+
                            str(datetime.datetime.now()))
        input('Press [Enter] to exit')


    
    def testing(self,data_params):

      display = DISPLAY().display


      # Import Params
      Data_Proc = Data_Process()
      params = Data_Proc.importer(data_params)

      display(True,True,params)

      # Initialize Data Processing
      plot_keys = dict_reorder(dict_reorder(params['process_params'],
                                              'plot_type',True))

      # Plot Data
      Data_Proc.plot_set(plot=params['kwargs']['plot'],keys=plot_keys)
      Data_Proc.plotter(
                    data=dict_reorder(params['process_params'],'data',True),
                    domain=dict_reorder(params['process_params'],'domain',True),
                    plot_props=params['plot_params'])      



    






if __name__ == '__main__':
    
    def grad(x_,y_,y_est):
      dydx = tf.gradients(y_est,x_)
      # print('Grad Shape',tf.shape(np.array*dydx))
      return tf.math.reduce_sum(dydx,axis=[0,-1])

    def sorter(v,i): 
      return array_sort(v,i,0,'ndarray')

    def data_seeder(seed_key, seed_val, seed_axis, seed_propor, 
                    new_type, seed_type,rand=False):
      data_shape = np.shape(seed_val)[seed_axis]
      if rand:
        return np.random.choice(data_shape,int(seed_propor*data_shape),
                              replace=False)
      else:
        return np.arange(data_shape-int(seed_propor*data_shape),data_shape)


    def output(y,x):
     return (np.array([np.reshape(np.take(y,i,1),(-1)) 
                    for i in range(np.shape(y)[1])],dtype=object),x)



    # Dataset Parameters
    data_params =  {'data_files': ['x_train','y_train'],
                                    # 'x_test','y_test','T_other'],
                    # 'data_sets': ['x_train','y_train','x_test','y_test',
                    #                 'T_other'],
                    'data_types': ['train','test','other'],
                    'data_format': 'npz',
                    'data_dir': 'dataset/',
                    'one_hot': [False,'y_'],
                    'upconvert': True,
                    'data_lists': True,
                    'data_obj_format':'array',
                    'data_seed':{ 'test':{'seed_type':'train',
                                  'seed_dimensions':[0,data_seeder,1/10],
                                  'seed_delim':'_',
                                  'remove_seed':True,
                                   'data_seed':True}}
                   }
    data_params_test = {'data_files': ['params'],
           'data_format': 'npz',
           'data_dir': 'dataset/neural_network/'}
           
    kwargs = {'train':True, 'test':True, 'other':True,
          'plot':{'plot_loop':False,'Training and Testing':True,
                  'Model Predictions':False,'Output':True},
          'save':True, 'printit':True, 'timeit':False}

    process_params = {'acc_train':{'domain':[],'data': [],
                                'data_name': 'Training\nAccuracy',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                        tf.reduce_mean(tf.abs((y_-y_est)/y_))},
                  'acc_test': { 'domain':[],'data': [],
                                'data_name': 'Testing\nAccuracy',
                                'domain_name': 'Epochs',
                                'data_type': 'test',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': lambda x_,y_,y_est: 
                                        tf.reduce_mean(tf.abs((y_-y_est)/y_))},
                  'output_train': { 'domain':[],'data': [],
                                'data_name': 'Output',
                                'domain_name': 'Time',
                                'data_type': 'train',
                                'domain_type': None,
                                'plot_type': 'Output',
                                'labels':['Training Label','Estimate'],
                                'data_wrapper':output,
                                'function': lambda x_,y_,y_est: 
                                              tf.stack([y_,y_est],1)[:,:,0]},
                  'output_test': { 'domain':[],'data': [],
                                'data_name': 'Output',
                                'domain_name': 'Time',
                                'data_type': 'test',
                                'domain_type': None,
                                'plot_type': 'Output',
                                'labels':['Testing Label','Estimate'],
                                'data_wrapper':output,
                                'function': lambda x_,y_,y_est: 
                                              tf.stack([y_,y_est],1)[:,:,0]},
                  'input_train': { 'domain':[],'data': [],
                                'data_name': 'Input',
                                'domain_name': 'Time',
                                'data_type': 'train',
                                'domain_type': None,
                                'plot_type': 'Output',
                                'labels':lambda x_,y_,y_est:
                                    ['Training_%d'%i 
                                    for i in range(np.shape(x_)[1])],
                                'data_wrapper':output,
                                'function': lambda x_,y_,y_est: 
                                    tf.stack([tf.slice(x_,[0,i],[-1,1]) 
                                    for i in range(np.shape(x_)[1])],1)[:,:,0]},              
                  'input_test': { 'domain':[],'data': [],
                                'data_name': 'Input',
                                'domain_name': 'Time',
                                'data_type': 'test',
                                'domain_type': None,
                                'plot_type': 'Output',
                                'labels':lambda x_,y_,y_est:
                                      ['Testing_%d'%i 
                                      for i in range(np.shape(x_)[1])],
                                'data_wrapper':output,
                                'function': lambda x_,y_,y_est: 
                                    tf.stack([tf.slice(x_,[0,i],[-1,1]) 
                                    for i in range(np.shape(x_)[1])],1)[:,:,0]},              
                  'cost': { 'domain':[],'data': [],
                                'data_name': 'Cost',
                                'domain_name': 'Epochs',
                                'data_type': 'train',
                                'domain_type': ['other','epochs'],
                                'plot_type': 'Training and Testing',
                                'labels':[],
                                'function': None},
                  # 'y_equiv':  { 'domain':[],'data': [],
                  #               'data_name': 'Average Output',
                  #               'domain_name': 'Temperature',
                  #               'data_type': 'test',
                  #               'domain_type': ['other','T_other'],
                  #               'plot_type': 'Model Predictions',
                  #               'labels':[],
                  #               'data_wrapper': lambda v,i: array_sort(v,i,0,
                  #                                       'ndarray'),
                  #               'function': lambda x_,y_,y_est: tf.equal(
                  #                                tf.argmax(y_est,axis=1),
                  #                                tf.argmax(y_,axis=1))},
                  # 'y_est':    { 'domain':[],'data': [],
                  #               'data_name': 'Phase Label',
                  #               'domain_name': 'Temperature',
                  #               'data_type': 'test',
                  #               'domain_type': ['other','T_other'],
                  #               'plot_type': 'Model Predictions',
                  #               'labels':['T<Tc','T>Tc'],
                  #               'data_wrapper': lambda v,i: array_sort(v,i,0,
                  #                                           'ndarray'),
                  #               'function': lambda x_,y_,y_est: y_est},
                  'y_grad':   { 'domain':[],'data': [],
                                'data_name': 'Output Gradient',
                                'domain_name': 'Energy',
                                'data_type': 'test',
                                'domain_type': ['test','y_test'],
                                'plot_type': 'Model Predictions',
                                'labels':[],
                                'data_wrapper': sorter,
                                'function': lambda x_,y_,y_est:
                                              grad(x_,y_,y_est)}
                                        # tf.math.reduce_sum(tf.gradients(
                                        #       y_est,x_),axis=[0,-1])}
                  }




    # Neural Network Parameters    
    alg_params = {
                  'n_neuron': [None,10,20,20,10,None], 
                  'alpha_learn': 0.0001, 'eta_reg': 0.0005,'sigma_var':0.1,                  
                  'n_epochs': 10,'n_batch_train': 1/5,'n_epochs_meas': 1/100,
                  'cost_func': 'mse_tf', 'optimize_func':'adam',
                  'regularize_func':'L2',
                  'method': 'neural_network_PhaseField',
                  'layers':'fcc' ,                
                  'neuron_func':{
                   'layer':  tf.nn.sigmoid,
                   'output': tf.nn.sigmoid
                              },                           
                  }
  
    # Plot Parameters
    plot_keys_special = ['y_est','output_train','output_test',
                            'input_test','input_train']
    plot_params = {
                     key: {
                    
                      'ax':   {'title' : '',
                               'xlabel':caps(process_params[key]['domain_name'],
                                            True,split_char='_') if (
                                            key in 
                                            ['cost','y_grad','input_test']) 
                                            else '',
                                'ylabel': caps(process_params[key]['data_name'],
                                                True,split_char='_'), 
                                },
                      'ax_attr': {'get_xticklabels':{'visible':True 
                                    if key in ['cost','input_test'] else False,
                                  'fontsize':12},
                                  'xaxis': {'ticks_position': 'none'},
                                  'get_yticklabels':{'visible':True,
                                  'fontsize':12}
                                  # 'set_xlabel':{'xlabel':key,'fontsize':12}
                                  },
                      'plot':  {'marker':'*' ,
                                'color':np.random.rand(3) if (
                                          key not in plot_keys_special) 
                                          else None},
                      
                      'data':  {'plot_type':'plot'},
                                
                      'other': {'label': lambda x='':caps(x,every_word=False,
                                                  sep_char=' ',split_char='_'),
                                'legend':True if key in plot_keys_special 
                                                    else False,
                                'legend': {'prop':{'size': 12}},
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
                                                    alg_params.items())))]),
                                              'x':0.925,'y':0.97,'ha':'left',
                                              'fontsize':7} if (
                                        key in ['acc_train','acc_test','cost'])
                                        else {'t':''},
                                  'pause':0.01
                                }
                     }
                    for key in process_params.keys()}
    
    
    
    # Run Neural Network 
    
    if 1:
      nn = optimizer()
      nn.training(alg_params,data_params,process_params,plot_params,**kwargs)
    else:
      nn_test = optimizer().testing(data_params_test)
    
