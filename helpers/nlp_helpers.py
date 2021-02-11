import torch as ch
import numpy as np

from lime.lime_text import LimeTextExplainer
from tqdm import tqdm
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

import os
from collections import defaultdict

def make_lime_fn(model,val_set, pooled_output, mu, std, bs=128): 
    device = 'cuda'
    def classifier_fn(sentences): 
        try:
            input_ids, attention_mask = zip(*[val_set.process_sentence(s) for s in sentences])
        except:
            input_ids, attention_mask = zip(*[val_set.dataset.process_sentence(s) for s in sentences])
        input_ids, attention_mask = ch.stack(input_ids), ch.stack(attention_mask)
        all_reps = []
        n = input_ids.size(0)
        # bs = args.batch_size
        for i in range(0,input_ids.size(0),bs): 
            i0 = min(i+bs,n)
            # reps, _ = 
            if hasattr(model, "roberta"): 
                output = model.roberta(input_ids=input_ids[i:i0].to(device), attention_mask=attention_mask[i:i0].to(device))
                output = output[0]

                # do RobertA classification head minus last out_proj classifier
                # https://huggingface.co/transformers/_modules/transformers/models/roberta/modeling_roberta.html
                output = output[:,0,:]
                output = model.classifier.dropout(output)
                output = model.classifier.dense(output)
                output = ch.tanh(output)
                cls_reps = model.classifier.dropout(output)
            else: 
                output = model.bert(input_ids=input_ids[i:i0].to(device), attention_mask=attention_mask[i:i0].to(device))
                if pooled_output: 
                    cls_reps = output[1]
                else:
                    cls_reps = output[0][:,0]
            cls_reps = cls_reps.cpu()
            cls_reps = (cls_reps - mu)/std
            all_reps.append(cls_reps.cpu())
        return ch.cat(all_reps,dim=0).numpy()
    return classifier_fn

def get_lime_features(model, val_set, val_loader, out_dir, pooled_output, mu, std): 
    os.makedirs(out_dir,exist_ok=True)
    explainer = LimeTextExplainer()
    reps_size = 768

    clf_fn = make_lime_fn(model,val_set, pooled_output, mu, std)
    files = []
    with ch.no_grad():
        print('number of sentences', len(val_loader))
        for i,(sentences, labels) in enumerate(tqdm(val_loader, total=len(val_loader), desc="Generating LIME")):
            assert len(sentences) == 1
            sentence, label = sentences[0], labels[0]
            if label.item() == -1: 
                continue
            out_file = f"{out_dir}/{i}.pth"
            try: 
                files.append(ch.load(out_file))
                continue
            except: 
                pass
            # if os.path.exists(out_file): 
            #   continue
            exp = explainer.explain_instance(sentence, clf_fn, labels=list(range(reps_size)))
            out = {
                "sentence": sentence, 
                "explanation": exp
            }
            ch.save(out, out_file)
            files.append(out)
    return files

def top_and_bottom_words(files, num_features=768): 
    top_words = []
    bot_words = []
    for j in range(num_features): 
        exps = [f['explanation'].as_list(label=j) for f in files]
        exps_collapsed = [a for e in exps for a  in e]

        accumulator = defaultdict(lambda: [])
        for word,weight in exps_collapsed: 
            accumulator[word].append(weight)
        exps_collapsed = [(k,np.array(accumulator[k]).mean()) for k in accumulator]


        exps_collapsed.sort(key=lambda a: a[1])

        weights = [a[1] for a in exps_collapsed]
        l = np.percentile(weights, q=1)
        u = np.percentile(weights, q=99)
        top_words.append([a for a in reversed(exps_collapsed) if a[1] > u])
        bot_words.append([a for a in exps_collapsed if a[1] < l])
    return top_words,bot_words

def get_explanations(feature_idxs, sparse, top_words, bot_words):
    expl_dict = {}
    maxfreq = 0
    for i,idx in enumerate(feature_idxs): 

        expl_dict[idx] = {}

        aligned = sparse[1,idx] > 0 
        for words, j in zip([top_words, bot_words],[0,1]): 
            if not aligned: 
                j = 1-j
            expl_dict[idx][j] = {
                a[0]:abs(a[1]) for a in words[idx] ## ARE THESE SIGNS CORRECT
            }
            maxfreq = max(maxfreq, max(list(expl_dict[idx][j].values())))

    for k in expl_dict:
        for s in expl_dict[k]:
            expl_dict[k][s] = {kk: vv / maxfreq for kk, vv in expl_dict[k][s].items()}
    return expl_dict

from matplotlib.colors import ListedColormap

def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    #cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    cmap = ListedColormap(sns.color_palette("RdYlGn_r").as_hex())
    fs = (font_size - 14) / (42 - 14)
    
    color_orig = cmap(fs)
    color = (255 * np.array(cmap(fs))).astype(np.uint8)

    return tuple(color[:3]) 

def plot_wordcloud(expln_dict, weights, factor=3, transpose=False, labels=("Positive Sentiment", "Negative Sentiment")):

    if transpose: 
        fig, axs = plt.subplots(2, len(expln_dict), figsize=(factor*3.5*len(expln_dict), factor*4), squeeze=False)
    else: 
        fig, axs = plt.subplots(len(expln_dict), 2, figsize=(factor*7, factor*1.5*len(expln_dict)), squeeze=False)
        
    for i,idx in enumerate(expln_dict.keys()): 
        if i == 0:
            if transpose: 
                axs[0,0].set_ylabel(labels[0], fontsize=36)
                axs[1,0].set_ylabel(labels[1], fontsize=36)
            else: 
                axs[i,0].set_title(labels[0], fontsize=36)
                axs[i,1].set_title(labels[1], fontsize=36)
    
        for j in [0, 1]: 
            wc = WordCloud(background_color="white", max_words=1000, min_font_size=14,
                           max_font_size=42)
            # generate word cloud
            d = {k[:20]:v for k,v in expln_dict[idx][j].items()}
            wc.generate_from_frequencies(d)
            default_colors = wc.to_array()
            
            if transpose: 
                ax = axs[j,i]
            else: 
                ax = axs[i,j]
            ax.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
                      interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            # ax.set_axis_off()
        if not transpose: 
            axs[i,0].set_ylabel(f"#{idx}\nW={weights[i]:.4f}", fontsize=36)
        else: 
            axs[0,i].set_title(f"#{idx}", fontsize=36)
    plt.tight_layout()
#     plt.subplots_adjust(left=None, bottom=0.1, right=None, top=0.9, wspace=0.25, hspace=0.0)
    return fig, axs