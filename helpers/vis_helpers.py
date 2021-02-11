import sys
sys.path.append('./lucent')
import torch as ch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import partial
from lucent.optvis import render, param, transform, objectives
from lime import lime_image

def plot_sparsity(results):
    """Function to visualize the sparsity-accuracy trade-off of regularized decision
    layers
    Args:
        results (dictionary): Appropriately formatted dictionary with regularization
        paths and logs of train/val/test accuracy.
    """
        
    if type(results['metrics']['acc_train'].values[0]) == list:
        all_tr = 100 * np.array(results['metrics']['acc_train'].values[0])
        all_val = 100 * np.array(results['metrics']['acc_val'].values[0])
        all_te = 100 * np.array(results['metrics']['acc_test'].values[0])
    else:
        all_tr = 100 * np.array(results['metrics']['acc_train'].values)
        all_val = 100 * np.array(results['metrics']['acc_val'].values)
        all_te = 100 * np.array(results['metrics']['acc_test'].values)

    fig, axarr = plt.subplots(1, 2, figsize=(14, 5))
    axarr[0].plot(all_tr)
    axarr[0].plot(all_val)
    axarr[0].plot(all_te)
    axarr[0].legend(['Train', 'Val', 'Test'], fontsize=16)
    axarr[0].set_ylabel("Accuracy (%)", fontsize=18)
    axarr[0].set_xlabel("Regularization index", fontsize=18)

    num_features = results['weights'][0].shape[1]
    total_sparsity = np.mean(results['sparsity'], axis=1) / num_features
    axarr[1].plot(total_sparsity, all_tr, 'o-')
    axarr[1].plot(total_sparsity, all_te, 'o-')
    axarr[1].legend(['Train', 'Val', 'Test'], fontsize=16)
    axarr[1].set_ylabel("Accuracy (%)", fontsize=18)
    axarr[1].set_xlabel("1 - Sparsity", fontsize=18)
    axarr[1].set_xscale('log')
    
    plt.show()

def normalize_weight(w):
    """Normalizes weights to a unit vector
    Args:
        w (tensor): Weight vector for a class.
    Returns:
        Normalized weight vector in the form of a numpy array.
    """
    return w.numpy() / np.linalg.norm(w.numpy())

def get_feature_visualization(model, feature_idx, signs):
    """Performs feature visualization using Lucid.
    Args:
        model: deep network whose deep features are to be visualized.
        feature_idx: indice of features to visualize.
        signs: +/-1 array indicating whether a feature should be maximized/minimized.
    Returns:
        Batch of feature visualizations .
    """
    param_f = lambda: param.image(224, batch=len(feature_idx), fft=True, decorrelate=True)
    obj = 0
    for fi, (f, s) in enumerate(zip(feature_idx, signs)):
        obj += s * objectives.channel('avgpool', f, batch=fi)
    op = render.render_vis(model.model, 
                           show_inline=False, 
                           objective_f=obj,
                           param_f=param_f, 
                           thresholds=(512,))[0]
    return ch.tensor(op).permute(0, 3, 1, 2)

def latent_predict(images, model, mean=None, std=None):
    """LIME helper function that computes the deep feature representation 
    for a given batch of images.
    Args:
        image (tensor): batch of images.
        model: deep network whose deep features are to be visualized.
        mean (tensor): mean of deep features.
        std (tensor): std deviation of deep features.
    Returns:
        Normalized deep features for batch of images.
    """
    preprocess_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])    
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    batch = ch.stack(tuple(preprocess_transform(i) for i in images), dim=0).to(device)
    
    (_, latents), _ = model(batch.to(ch.float), with_latent=True)
    scaled_latents = (latents.detach().cpu() - mean.to(ch.float)) / std.to(ch.float)
    return scaled_latents.numpy()

def parse_lime_explanation(expln, f, sign, NLime=3):
    """LIME helper function that extracts a mask from a lime explanation
    Args:
        expln: LIME explanation from LIME library
        f: indice of features to visualize
        sign: +/-1 array indicating whether the feature should be maximized/minimized.
        images (tensor): batch of images.
        NLime (int): Number of top-superpixels to visualize.
    Returns:
        Tensor where the first and second channels contains superpixels that cause the
        deep feature to activate and deactivate respectively.
    """
    segs = expln.segments
    vis_mask = np.zeros(segs.shape + (3,))

    weights = sorted([v for v in expln.local_exp[f]], 
                     key=lambda x: -np.abs(x[1]))
    weight_values = [w[1] for w in weights]
    pos_lim, neg_lim = np.max(weight_values), (1e-8 + np.min(weight_values))

    if NLime is not None:
        weights = weights[:NLime]

    for wi, w in enumerate(weights):
        if w[1] >= 0:
            si = (w[1] / pos_lim, 0, 0) if sign == 1 else (0, w[1] / pos_lim, 0)
        else:
            si = (0, w[1] / neg_lim, 0) if sign == 1 else (w[1] / neg_lim, 0, 0) 
        vis_mask[segs == w[0]] = si

    return ch.tensor(vis_mask.transpose(2, 0, 1))

def get_lime_explanation(model, feature_idx, signs,
                         images, rep_mean, rep_std,  
                         num_samples=1000,
                         NLime=3,
                         background_color=0.6):
    """Computes LIME explanations for a given set of deep features. The LIME
    objective in this case is to identify the superpixels within the specified
    images that maximally/minimally activate the corresponding deep feature.
    Args:
        model: deep network whose deep features are to be visualized.
        feature_idx: indice of features to visualize
        signs: +/-1 array indicating whether a feature should be maximized/minimized.
        images (tensor): batch of images.
        rep_mean (tensor): mean of deep features.
        rep_std (tensor): std deviation of deep features.
        NLime (int): Number of top-superpixels to visualize
        background_color (float): Color to assign non-relevant super pixels
    Returns:
        Tensor comprising LIME explanations for the given set of deep features.
    """
    explainer = lime_image.LimeImageExplainer()
    lime_objective = partial(latent_predict, model=model, mean=rep_mean, std=rep_std)

    explanations = []
    for im, feature, sign in zip(images, feature_idx, signs):
        explanation = explainer.explain_instance(im.numpy().transpose(1, 2, 0), 
                                     lime_objective, 
                                     labels=np.array([feature]), 
                                     top_labels=None,
                                     hide_color=0, 
                                     num_samples=num_samples) 
        explanation = parse_lime_explanation(explanation, 
                                             feature, 
                                             sign, 
                                             NLime=NLime)
        
        if sign == 1:
            explanation = explanation[:1].unsqueeze(0).repeat(1, 3, 1, 1)
        else:
            explanation = explanation[1:2].unsqueeze(0).repeat(1, 3, 1, 1)
        
        interpolated = im * explanation + background_color * ch.ones_like(im) * (1 - explanation)
        explanations.append(interpolated)
        
    return ch.cat(explanations)