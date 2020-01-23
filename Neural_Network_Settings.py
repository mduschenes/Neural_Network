# Neural Network Settings

def grad(x_,y_,y_est):
      dydx = tf.gradients(y_est,x_)
      # print('Grad Shape',tf.shape(np.array*dydx))
      return tf.math.reduce_sum(dydx,axis=[0,-1])

    def sorter(v,i): 
      return array_sort(v,i,0,'ndarray')


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
                                  'seed_dimensions':[0,np.random.choice(1125,
                                                        500,replace=False)],
                                  'seed_delim':'_',
                                  'remove_seed':True,
                                   'data_seed':True}}
                   }
    kwargs = {'train':True, 'test':True, 'other':True,
          'plot':{'plot_loop':False,'Training and Testing':True,
                  'Model Predictions':True,'Gradients':False},
          'save':True, 'printit':False, 'timeit':False}

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
                                'data_wrapper': lambda v,i: array_sort(v,i,0,
                                                        'ndarray'),
                                'function': lambda x_,y_,y_est:
                                              grad(x_,y_,y_est)}
                                        # tf.math.reduce_sum(tf.gradients(
                                        #       y_est,x_),axis=[0,-1])}
                  }




    # Neural Network Parameters    
    network_params = {
                  'n_neuron': [None,100,None], 
                  'alpha_learn': 0.0035, 'eta_reg': 0.0005,'sigma_var':0.1,                  
                  'n_epochs': 50,'n_batch_train': 1/10,'n_epochs_meas': 1/50,
                  'cost_func': 'cross_entropy', 'optimize_func':'adam',
                  'regularize':'L2',
                  'method': 'neural_network_PhaseField',
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
    