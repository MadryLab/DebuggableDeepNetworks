import os, sys
sys.path.append('..')
import numpy as np
import torch as ch
from torch.utils.data import TensorDataset
from robustness.datasets import DATASETS as VISION_DATASETS
from robustness.tools.label_maps import CLASS_DICT
from language.datasets import DATASETS as LANGUAGE_DATASETS
from language.models import LANGUAGE_MODEL_DICT
from transformers import AutoTokenizer

def get_label_mapping(dataset_name):
    if dataset_name == 'imagenet':
        return CLASS_DICT['ImageNet']
    elif dataset_name == 'places-10':
        return CD_PLACES
    elif dataset_name == 'sst':
        return {0: 'negative', 1: 'positive'} 
    elif 'jigsaw' in dataset_name:
        category = dataset_name.split('jigsaw-')[1] if 'alt' not in dataset_name \
                else dataset_name.split('jigsaw-alt-')[1]
        return {0: f'not {category}', 1: f'{category}'} 
    else:
        raise ValueError("Dataset not currently supported...")

def load_dataset(dataset_name, dataset_path, dataset_type,
                  batch_size, num_workers, 
                  maxlen_train=256, maxlen_val=256, 
                  shuffle=False, model_path=None, return_sentences=False):
    
    
    if dataset_type == 'vision': 
        if dataset_name == 'places-10': dataset_name = 'places365'        
        if dataset_name not in VISION_DATASETS:
            raise ValueError("Vision dataset not currently supported...")
        dataset = VISION_DATASETS[dataset_name](os.path.expandvars(dataset_path))
        
        if dataset_name == 'places365': 
            dataset.num_classes = 10
        
        train_loader, test_loader = dataset.make_loaders(num_workers, 
                                                        batch_size, 
                                                        data_aug=False, 
                                                        shuffle_train=shuffle, 
                                                        shuffle_val=shuffle)
        return dataset, train_loader, test_loader
    else:
        if model_path is None:
            model_path = LANGUAGE_MODEL_DICT[dataset_name]
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        kwargs = {} if 'jigsaw' not in dataset_name else \
                    {'label': dataset_name[11:] if 'alt' in dataset_name \
                        else dataset_name[7:]}
        kwargs['return_sentences'] = return_sentences
        train_set = LANGUAGE_DATASETS(dataset_name)(filename=f'{dataset_path}/train.tsv', 
                                                    maxlen=maxlen_train, 
                                                    tokenizer=tokenizer,
                                                    **kwargs)
        test_set = LANGUAGE_DATASETS(dataset_name)(filename=f'{dataset_path}/test.tsv', 
                                                  maxlen=maxlen_val, 
                                                  tokenizer=tokenizer,
                                                  **kwargs)
        train_loader = ch.utils.data.DataLoader(dataset=train_set, 
                                                batch_size=batch_size, 
                                                num_workers=num_workers)
        test_loader = ch.utils.data.DataLoader(dataset=test_set, 
                                               batch_size=batch_size, 
                                               num_workers=num_workers)
        #assert len(np.unique(train_set.df['label'].values)) == len(np.unique(test_set.df['label'].values))
        train_set.num_classes = 2
        # train_loader.dataset.targets = train_loader.dataset.df['label'].values
        # test_loader.dataset.targets = test_loader.dataset.df['label'].values
        
        return train_set, train_loader, test_loader


class IndexedTensorDataset(ch.utils.data.TensorDataset): 
    def __getitem__(self, index): 
        val = super(IndexedTensorDataset, self).__getitem__(index)
        return val + (index,)
    
class IndexedDataset(ch.utils.data.Dataset): 
    def __init__(self, ds, sample_weight=None): 
        super(ch.utils.data.Dataset, self).__init__()
        self.dataset = ds
        self.sample_weight=sample_weight
    
    def __getitem__(self, index): 
        val = self.dataset[index]
        if self.sample_weight is None: 
            return val + (index,)
        else: 
            weight = self.sample_weight[index]
            return val + (weight,index)
    def __len__(self): 
        return len(self.dataset)

def add_index_to_dataloader(loader, sample_weight=None): 
    return ch.utils.data.DataLoader(
        IndexedDataset(loader.dataset, sample_weight=sample_weight), 
        batch_size=loader.batch_size, 
        sampler=loader.sampler, 
        num_workers=loader.num_workers, 
        collate_fn=loader.collate_fn, 
        pin_memory=loader.pin_memory, 
        drop_last=loader.drop_last, 
        timeout=loader.timeout, 
        worker_init_fn=loader.worker_init_fn, 
        multiprocessing_context=loader.multiprocessing_context
    )

class NormalizedRepresentation(ch.nn.Module): 
    def __init__(self, loader, metadata, device='cuda', tol=1e-5): 
        super(NormalizedRepresentation, self).__init__()


        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = ch.clamp(metadata['X']['std'], tol)

    def forward(self, X): 
        return (X - self.mu.to(self.device))/self.sigma.to(self.device)
    
CD_PLACES = {0: 'airport_terminal',
 1: 'boat_deck',
 2: 'bridge',
 3: 'butchers_shop',
 4: 'church-outdoor',
 5: 'hotel_room',
 6: 'laundromat',
 7: 'river',
 8: 'ski_slope',
 9: 'volcano'}
