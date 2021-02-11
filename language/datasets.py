import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig

from .jigsaw_loaders import JigsawDataOriginal

def DATASETS(dataset_name):
    if dataset_name == 'sst': return SSTDataset
    elif dataset_name.startswith('jigsaw'): return JigsawDataset
    else:
        raise ValueError("Language dataset is not currently supported...")

class SSTDataset(Dataset):
	"""
	Stanford Sentiment Treebank V1.0
	Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank
	Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher Manning, Andrew Ng and Christopher Potts
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2013)
	"""
	def __init__(self, filename, maxlen, tokenizer, return_sentences=False): 
		#Store the contents of the file in a pandas dataframe
		self.df = pd.read_csv(filename, delimiter = '\t')
		#Initialize the tokenizer for the desired transformer model
		self.tokenizer = tokenizer
		#Maximum length of the tokens list to keep all the sequences of fixed size
		self.maxlen = maxlen
		#whether to tokenize or return raw setences
		self.return_sentences = return_sentences

	def __len__(self):
		return len(self.df)

	def __getitem__(self, index):    
		#Select the sentence and label at the specified index in the data frame
		sentence = self.df.loc[index, 'sentence']
		label = self.df.loc[index, 'label']
		#Preprocess the text to be suitable for the transformer
		if self.return_sentences: 
			return sentence, label
		else: 
			input_ids, attention_mask = self.process_sentence(sentence)
			return input_ids, attention_mask, label

	def process_sentence(self, sentence): 
		tokens = self.tokenizer.tokenize(sentence) 
		tokens = ['[CLS]'] + tokens + ['[SEP]'] 
		if len(tokens) < self.maxlen:
			tokens = tokens + ['[PAD]' for _ in range(self.maxlen - len(tokens))] 
		else:
			tokens = tokens[:self.maxlen-1] + ['[SEP]'] 
		#Obtain the indices of the tokens in the BERT Vocabulary
		input_ids = self.tokenizer.convert_tokens_to_ids(tokens) 
		input_ids = torch.tensor(input_ids) 
		#Obtain the attention mask i.e a tensor containing 1s for no padded tokens and 0s for padded ones
		attention_mask = (input_ids != 0).long()
		return input_ids, attention_mask

class JigsawDataset(Dataset):
	def __init__(self, filename, maxlen, tokenizer, return_sentences=False, label="toxic"): 
		classes=[label]
		if 'train' in filename: 
			self.dataset = JigsawDataOriginal(
				train_csv_file=filename, 
				test_csv_file=None, 
				train=True, 
				create_val_set=False,
				add_test_labels=False, 
				classes=classes
			)
		elif 'test' in filename: 
			self.dataset = JigsawDataOriginal(
				train_csv_file=None, 
				test_csv_file=filename, 
				train=False, 
				create_val_set=False,
				add_test_labels=True, 
				classes=classes
			)
		else:
			raise ValueError("Unknown filename {filename}")
		# #Store the contents of the file in a pandas dataframe
		# self.df = pd.read_csv(filename, header=None, names=['label', 'sentence'])
		#Initialize the tokenizer for the desired transformer model
		self.tokenizer = tokenizer
		#Maximum length of the tokens list to keep all the sequences of fixed size
		self.maxlen = maxlen
		#whether to tokenize or return raw setences
		self.return_sentences = return_sentences

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, index):    
		#Select the sentence and label at the specified index in the data frame
		sentence, meta = self.dataset[index]
		label = meta["multi_target"].squeeze()

		#Preprocess the text to be suitable for the transformer
		if self.return_sentences: 
			return sentence, label
		else: 
			input_ids, attention_mask = self.process_sentence(sentence)
			return input_ids, attention_mask, label

	def process_sentence(self, sentence): 
		# print(sentence)
		d = self.tokenizer(sentence, padding='max_length', truncation=True)
		input_ids = torch.tensor(d["input_ids"])
		attention_mask = torch.tensor(d["attention_mask"])
		return input_ids, attention_mask