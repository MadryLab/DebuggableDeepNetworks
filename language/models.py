import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel, BertForSequenceClassification, RobertaForSequenceClassification

LANGUAGE_MODEL_DICT = {
    'sst': 'barissayil/bert-sentiment-analysis-sst',
    'jigsaw-toxic': 'unitary/toxic-bert', 
    'jigsaw-severe_toxic': 'unitary/toxic-bert', 
    'jigsaw-obscene': 'unitary/toxic-bert', 
    'jigsaw-threat': 'unitary/toxic-bert', 
    'jigsaw-insult': 'unitary/toxic-bert', 
    'jigsaw-identity_hate': 'unitary/toxic-bert', 
    'jigsaw-alt-toxic': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-severe_toxic': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-obscene': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-threat': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-insult': 'unitary/unbiased-toxic-roberta', 
    'jigsaw-alt-identity_hate': 'unitary/unbiased-toxic-roberta'
}

class BertForSentimentClassification(BertPreTrainedModel):
	def __init__(self, config):
		super().__init__(config)
		self.bert = BertModel(config)
		# self.dropout = nn.Dropout(config.hidden_dropout_prob)
		#The classification layer that takes the [CLS] representation and outputs the logit
		self.cls_layer = nn.Linear(config.hidden_size, 1)

	def forward(self, input_ids, attention_mask):
		'''
		Inputs:
			-input_ids : Tensor of shape [B, T] containing token ids of sequences
			-attention_mask : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
			(where B is the batch size and T is the input length)
		'''
		#Feed the input to Bert model to obtain contextualized representations
		reps, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
		#Obtain the representations of [CLS] heads
		cls_reps = reps[:, 0]
		# cls_reps = self.dropout(cls_reps)
		logits = self.cls_layer(cls_reps)
		return logits