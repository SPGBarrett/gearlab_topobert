#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  backbone_models.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2022/5/23 19:12
@Version    :  1.0
@Description:  Customized BERT models for token classification.

Example:
    Some examples of usage here.
Attributes:
    Attribute description here.
Todo:
    * For module TODOs

'''

# import lib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup,
                          BertPreTrainedModel, BertModel)
from torch import nn
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report

#################################################################################
#                                                                               #
#                              Public Parameters                                #
#                                                                               #
#################################################################################
_model_defualt_args = {
    "--cuda": "use GPU",
    "--pretrained_model": "bert-base-cased",
    "--num_of_labels": 11,
    "--model_hidden_layer_size": 768,
    "--no_hidden_layers": 13,
    "--dropout": 0.1,
    "--out-channel": 16,
    "--freeze-bert": False,
    "--verbose": "whether to output the test results"
}

#################################################################################
#                                                                               #
#                              Customized Models                                #
#                                                                               #
#################################################################################

########################################################################################
#                              BERT + Linear Classifier V1                             #
########################################################################################
class BertSimpleNer(BertForTokenClassification):
    '''
        Bert NER classification model inherited from BertForTokenClassification
        Using a simple linear classifier stacked on BERT
    '''

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size, max_len, feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

########################################################################################
#                         BERT + CNN on Hidden State + Reshape 2D                      #
########################################################################################
class BertCNN2DNer(BertModel):

    def __init__(self, bert_config=None, model_config=_model_defualt_args):
        """
        :param bert_config: str, BERT configuration description
        """
        self.model_config = model_config
        if bert_config is None:
            self.config = BertConfig.from_pretrained(self.model_config['--pretrained_model'],
                                                     num_labels=self.model_config['--num_of_labels'])
        else:
            self.config = bert_config

        super(BertCNN2DNer, self).__init__(self.config)

        # Init parameter:
        self.dropout_rate = self.model_config['--dropout']
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.num_labels = self.model_config['--num_of_labels']
        self.in_channel = self.model_config['--no_hidden_layers']
        self.out_channel = self.model_config['--out-channel']

        # Language Model: Bert inherit from BertForTokenClassification
        self.bert = BertModel.from_pretrained(self.model_config['--pretrained_model'], add_pooling_layer=False, config=self.config)

        # Tokenizer:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config['--pretrained_model'])

        # Classifier structure:
        # Convolution Layer 1:
        self.conv1 = nn.Conv2d(in_channels=self.in_channel,
                              out_channels=self.out_channel,
                              kernel_size=(3, 3),
                              groups=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(in_channels=self.out_channel,
                               out_channels= int((self.out_channel/2)),
                               kernel_size=(2, 2),
                               groups=1)
        self.fc_input_size = int(self.out_channel/2*5*7) # Calculate according to reshaped size
        # Fully Connected Layer 1:
        self.fc1 = nn.Linear(self.fc_input_size, 128, bias=True)
        # Fully Connected Layer 2: also the final layer
        self.fc_out = nn.Linear(128, self.num_labels)

        # Freeze the BERT model: Feature-based training approach
        if self.model_config['--freeze-bert']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, valid_ids=None, attention_mask_label=None):
        # Process original output (Not in use)
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None, output_hidden_states=True)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        # Encode with BERT and pull hidden layer and the final layer out as the convolution NN input:
        bert_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None,
                                output_hidden_states=True)  # Should output hidden layer
        # Unpack data
        hidden_state_tuple = bert_output['hidden_states']
        total_stack_layer = torch.stack(hidden_state_tuple, 2)  # dim(batch_size, max_seq, layer_num, hidden_layer_size)
        # Reshape for better CNN: (The trick and novelty)
        batch_size, max_seq, layer_number, hidden_layer_size = total_stack_layer.shape
        total_reshape = torch.reshape(total_stack_layer, [batch_size * max_seq, 13, 32, 24])

        # Convolution and max pooling
        nn_out = self.conv1(total_reshape)  # (batch_size, channel_out, some_length)
        nn_out = self.pool(nn_out)
        nn_out = self.conv2(nn_out)  # (batch_size, channel_out, some_length)
        nn_out = self.pool(nn_out)
        nn_out = torch.reshape(nn_out, [batch_size, max_seq, 5 * 7 * 16])
        # Fully connected and activation:
        nn_out = self.fc1(nn_out)
        nn_out = F.relu(nn_out)
        nn_out = self.fc_out(nn_out)

        # Get active parts and calculate loss:
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = nn_out.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(nn_out.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn_out


########################################################################################
#                         BERT + CNN on Output State + 1D Signal                       #
########################################################################################
class BertCNN1DNer(BertModel):

    def __init__(self, bert_config=None, model_config=_model_defualt_args):
        '''
        Args:
            bert_config (BertConfig): The huggingface bert config object. Will be passed through by from_pretrained() function
            model_config (json): The customized parameter required to construct the model. When loading the model from
            existing parameters, read file and pass the result to the from_pretrained function using key 'model_config'
            freeze_bert (bool): Whether using feature-based or fine-tuning while training.
        '''
        self.model_config = model_config
        if bert_config is None:
            self.config = BertConfig.from_pretrained(self.model_config['--pretrained_model'],
                                                     num_labels=self.model_config['--num_of_labels'])
        else:
            self.config = bert_config

        super(BertCNN1DNer, self).__init__(self.config)

        # Init parameter:
        self.dropout_rate = self.model_config['--dropout']
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.num_labels = int(self.model_config['--num_of_labels'])
        self.in_channel = int(self.model_config['--no_hidden_layers'])
        self.out_channel = int(self.model_config['--out-channel'])

        # Language Model: Bert inherit from BertForTokenClassification
        self.bert = BertModel.from_pretrained(self.model_config['--pretrained_model'], add_pooling_layer=False,
                                              config=self.config)

        # Tokenizer:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config['--pretrained_model'])

        # Classifier structure:
        # Convolution Layer 1:
        self.conv1 = nn.Conv1d(in_channels=1,
                              out_channels=self.out_channel,
                              kernel_size=3,
                              groups=1)
        self.pool = nn.MaxPool1d(2)
        self.vector_size = int((model_config['--model_hidden_layer_size'] - 3 + 1) / 2)
        # Fully Connected Layer 1:
        self.fc1 = nn.Linear(int(self.out_channel * self.vector_size), 128, bias=True)
        # Fully Connected Layer 2: also the final layer
        self.fc_out = nn.Linear(128, self.num_labels)

        # Freeze the BERT model: Feature-based training approach
        if self.model_config['--freeze-bert']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        # Process the output of bert output state
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None, output_hidden_states=True)[0]
        batch_size, max_seq_len, hidden_layer_size = sequence_output.shape
        valid_output = torch.zeros(batch_size, max_seq_len, hidden_layer_size, dtype=torch.float32, device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_seq_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        # Dropout:
        sequence_output = self.dropout(valid_output)
        # Reshape output for CNN:
        total_reshape = torch.reshape(sequence_output, [batch_size * max_seq_len, 1, hidden_layer_size])

        # Convolution and max pooling
        nn_out = self.conv1(total_reshape)  # (batch_size, channel_out, some_length)
        nn_out = self.pool(nn_out)
        nn_out = torch.reshape(nn_out, [batch_size, max_seq_len, self.out_channel * self.vector_size])
        # Fully connected and activation:
        nn_out = self.fc1(nn_out)
        nn_out = F.relu(nn_out)
        nn_out = self.fc_out(nn_out)

        # Get active parts and calculate loss:
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = nn_out.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fn(active_logits, active_labels)
            else:
                loss = loss_fn(nn_out.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return nn_out



class BertMlpNer(BertForTokenClassification):

    def __init__(self, bert_config=None, model_config=_model_defualt_args):
        '''
        Args:
            bert_config ():
            freeze_bert (): Feature-Based or Fine-tuning
        '''
        self.model_config = model_config
        if bert_config is None:
            self.config = BertConfig.from_pretrained(self.model_config['--pretrained_model'],
                                                     num_labels=self.model_config['--num_of_labels'])
        else:
            self.config = bert_config

        super(BertMlpNer, self).__init__(self.config)
        # Init parameter:
        self.num_labels = self.model_config['--num_of_labels']

        # Language Model: Bert inherit from BertForTokenClassification
        self.bert = BertModel.from_pretrained(self.model_config['--pretrained_model'], add_pooling_layer=False, config=self.config)

        # Tokenizer:
        self.tokenizer = BertTokenizer.from_pretrained(self.model_config['--pretrained_model'])

        # Classifier structure:
        # Fully Connected Layer 1:
        self.fc1 = nn.Linear(model_config['--model_hidden_layer_size'], 256, bias=True)
        # Fully Connected Layer 2: also the final layer
        self.fc_out = nn.Linear(256, self.num_labels)

        # Freeze the BERT model: Feature-based training approach
        if self.model_config['--freeze-bert']:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None, attention_mask_label=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32,device='cuda')
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                    if valid_ids[i][j].item() == 1:
                        jj += 1
                        valid_output[i][jj] = sequence_output[i][j]
        # Dropout:
        sequence_output = self.dropout(valid_output)
        # MLP:
        logits = self.fc1(sequence_output)
        logits = F.relu(logits)
        logits = self.fc_out(logits)

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            #attention_mask_label = None
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits



def main():
    return

if __name__ == '__main__':
    main()