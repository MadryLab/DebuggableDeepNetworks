import os
import numpy as np
import torch as ch
import pandas as pd

def load_glm(result_dir):
    
    Nlambda = max([int(f.split('params')[1].split('.pth')[0]) 
                   for f in os.listdir(result_dir) if 'params' in f]) + 1
    
    print(f"Loading regularization path of length {Nlambda}")
    
    params_dict = {i: ch.load(os.path.join(result_dir, f"params{i}.pth"),
                          map_location=ch.device('cpu')) for i in range(Nlambda)}
    
    regularization_strengths = [params_dict[i]['lam'].item() for i in range(Nlambda)]
    weights = [params_dict[i]['weight'] for i in range(Nlambda)]
    biases = [params_dict[i]['bias'] for i in range(Nlambda)]
   
    metrics = {'acc_tr': [], 'acc_val': [], 'acc_test': []}
    
    for k in metrics.keys():
        for i in range(Nlambda):
            metrics[k].append(params_dict[i]['metrics'][k])
        metrics[k] = 100 * np.stack(metrics[k])
    metrics = pd.DataFrame(metrics)
    metrics = metrics.rename(columns={'acc_tr': 'acc_train'})
    
    weights_stacked = ch.stack(weights)
    sparsity = ch.sum(weights_stacked != 0, dim=2).numpy()
            
    return {'metrics': metrics, 
            'regularization_strengths': regularization_strengths, 
            'weights': weights, 
            'biases': biases,
            'sparsity': sparsity,
            'weight_dense': weights[-1],
            'bias_dense': biases[-1]}

def select_sparse_model(result_dict, 
                        selection_criterion='absolute', 
                        factor=6):
    
    assert selection_criterion in ['sparsity', 'absolute', 'relative', 'percentile'] 

    metrics, sparsity = result_dict['metrics'], result_dict['sparsity']
    
    acc_val, acc_test = metrics['acc_val'], metrics['acc_test']

    if factor == 0:
        sel_idx = -1
    elif selection_criterion == 'sparsity':
        sel_idx = np.argmin(np.abs(np.mean(sparsity, axis=1) - factor))
    elif selection_criterion == 'relative':
        sel_idx = np.argmin(np.abs(acc_val - factor * np.max(acc_val)))
    elif selection_criterion == 'absolute':
        delta = acc_val - (np.max(acc_val) - factor)
        lidx = np.where(delta <= 0)[0]
        sel_idx = lidx[np.argmin(-delta[lidx])]
    elif selection_criterion == 'percentile': 
        diff = np.max(acc_val) - np.min(acc_val)
        sel_idx = np.argmax(acc_val >  np.max(acc_val) - factor * diff)

    print(f"Test accuracy | Best: {max(acc_test): .2f},",
          f"Sparse: {acc_test[sel_idx]:.2f}",
          f"Sparsity: {np.mean(sparsity[sel_idx]):.2f}")

    result_dict.update({'weight_sparse': result_dict['weights'][sel_idx], 
                        'bias_sparse': result_dict['biases'][sel_idx]})
    return result_dict