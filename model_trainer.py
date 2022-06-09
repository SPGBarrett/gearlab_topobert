#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  model_trainer.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2022/5/24 16:01
@Version    :  1.0
@Description:  None.

Example:
    Some examples of usage here.
Attributes:
    Attribute description here.
Todo:
    * For module TODOs

'''

# import lib
import random
import numpy as np
import torch
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import TensorDataset, RandomSampler
import torch.nn.functional as F
from transformers import (WEIGHTS_NAME, AdamW, BertConfig,
                          BertForTokenClassification, BertTokenizer, get_linear_schedule_with_warmup,
                          BertPreTrainedModel, BertModel)
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange
from datetime import datetime
import os
import json
from seqeval.metrics import classification_report
from bzrs_main.modules.ml_models.topo_bert.dataset_process import *
from bzrs_main.modules.ml_models.topo_bert.backbone_models import *

logger = logging.getLogger(__name__)

# Training configurations:
_default_train_config = {
    "--task_name": "bert_geoparsing",
    "--toponym_only": False,
    "--random_seed": 42,
    "--use_gpu": 1,
    "--train_data_type": "conll",
    "--validate_data_type": "conll",
    "--test_data_type": "conll",
    "--train_data_dir": "",
    "--validate_data_dir": "",
    "--test_data_dir": "",
    "--train_data_file": "",
    "--validate_data_file": "",
    "--test_data_file": "",
    "--is_validate": 1,
    "--is_test": 1,
    "--output_dir": "",
    "--cache_dir": "",
    "--bert_model": "bert-base-uncased",
    "--do_lower_case": True,
    "--max_seq_length": 128,
    "--training_epoch": 40,
    "--train_batch_size": 4,
    "--test_batch_size": 4,
    "--learning_rate": 5e-5,
    "--warm_up_proportion": 0.1,
    "--weight_decay": 0.01,
    "--adam_epsilon": 1e-8,
    "--max_grad_norm": 1.0,
    "--num_grad_accum_steps": 1,
    "--loss_scale": 0
}


_demo_train_config = {
    "--task_name": "bert_geoparsing",
    "--toponym_only": False,
    "--random_seed": 42,
    "--use_gpu": 1,
    "--train_data_type": "conll",
    "--validate_data_type": "conll",
    "--test_data_type": "conll",
    "--train_data_dir": "E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
    "--validate_data_dir": "E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
    "--test_data_dir": "E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
    "--train_data_file": "train.txt",
    "--validate_data_file": "test.txt",
    "--test_data_file": "test.txt",
    "--is_validate": 1,
    "--is_test": 1,
    "--output_dir": "./outputs",
    "--cache_dir": "./cache",
    "--bert_model": "bert-base-cased",
    "--do_lower_case": False,
    "--max_seq_length": 128,
    "--training_epoch": 40,
    "--train_batch_size": 4,
    "--test_batch_size": 4,
    "--learning_rate": 5e-5,
    "--warm_up_proportion": 0.1,
    "--weight_decay": 0.01,
    "--adam_epsilon": 1e-8,
    "--max_grad_norm": 1.0,
    "--num_grad_accum_steps": 1,
    "--loss_scale": 0
}

data_processer_map = {
    "conll": Processor_CoNLL2003,
    "wiki": Processor_Standard_V1,
    "wnut": Processor_Standard_V1,
    "harvey": Processor_Harvey
}


class TopoBertModelTrainer():

    def __init__(self, model, tokenizer = None, train_config = _default_train_config):
        self.train_config = train_config
        self.model = model
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained(self.train_config['--bert_model'],
                                                  do_lower_case=self.train_config['--do_lower_case'])
        else:
            self.tokenizer = tokenizer
        # init training and testing data:
        # Get data processor:
        self.train_data_processors = data_processer_map[self.train_config['--train_data_type']]()
        self.train_examples = self.train_data_processors.get_train_data(self.train_config['--train_data_dir'],
                                                              self.train_config['--train_data_file'],
                                                              self.train_config['--toponym_only'])
        self.test_data_processors = data_processer_map[self.train_config['--test_data_type']]()
        self.eval_examples = self.test_data_processors.get_dev_data(self.train_config['--test_data_dir'],
                                                              self.train_config['--test_data_file'],
                                                              self.train_config['--toponym_only'])
        self.current_output_dir = None


    def customize_datasets(self, training_examples, testing_examples, train_data_processors=None, test_data_processors=None):
        '''
            Using customized training and testing data
        Args:
            training_examples ():
            testing_examples ():
            train_data_processors ():
            test_data_processors ():

        Returns:

        '''
        if train_data_processors is not None:
            self.train_data_processors = train_data_processors
        if test_data_processors is not None:
            self.test_data_processors = test_data_processors
        self.train_examples = training_examples
        self.eval_examples = testing_examples


    def train(self):
        # Initial checks:
        # Check number of gradient accumulation steps:
        if self.train_config['--num_grad_accum_steps'] < 1:
            raise ValueError(
                "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                    self.train_config['--num_grad_accum_steps']))
        # Check model output directory
        if not os.path.exists(self.train_config['--output_dir']):
            print(f"Creating output dir: {self.train_config['--output_dir']}")
            os.makedirs(self.train_config['--output_dir'])

        # Get device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'CUDA Available: {torch.cuda.is_available()}')
        print(device)
        # Set seed:
        random.seed(self.train_config['--random_seed'])
        np.random.seed(self.train_config['--random_seed'])
        torch.manual_seed(self.train_config['--random_seed'])
        # Calculate training batch size:
        train_batch_size = self.train_config['--train_batch_size'] // self.train_config['--num_grad_accum_steps']
        # Get task name:
        task_name = self.train_config['--task_name'].lower()
        label_list = self.train_data_processors.get_labels(filter_toponym=self.train_config['--toponym_only'])
        num_labels = len(label_list) + 1

        num_train_optimization_steps = 0
        # Get training data:
        num_train_optimization_steps = int(
            len(self.train_examples) / train_batch_size / self.train_config['--num_grad_accum_steps'] * self.train_config['--training_epoch'])


        # Load model to cuda device:
        self.model.to(device)

        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.train_config['--weight_decay']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        warmup_steps = int(self.train_config['--warm_up_proportion'] * num_train_optimization_steps)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.train_config['--learning_rate'], eps=self.train_config['--adam_epsilon'])
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1)

        global_step = 0
        nb_tr_steps = 0
        tr_loss = 0
        label_map = {i: label for i, label in enumerate(label_list, 1)}

        # Start training:
        train_features = convert_examples_to_features(
            self.train_examples, label_list, self.train_config['--max_seq_length'], self.tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_examples))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        # Construct data:
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in train_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                   all_lmask_ids)

        train_sampler = RandomSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

        self.model.train()

        for _ in trange(int(self.train_config['--training_epoch']), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask = batch
                loss = self.model(input_ids, segment_ids, input_mask, label_ids, valid_ids, l_mask)

                if self.train_config['--num_grad_accum_steps'] > 1:
                    loss = loss / self.train_config['--num_grad_accum_steps']

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_config['--max_grad_norm'])

                tr_loss += loss.item()
                print(loss.item())
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                self.model.zero_grad()
                global_step += 1

        # Save a trained model and the associated configuration
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  # Only save the model it-self
        current_datetime = datetime.now()
        current_datetime_dir = current_datetime.strftime("%Y%m%d%H%M%S") + '\\'
        model_output_dir = os.path.join(self.train_config['--output_dir'], current_datetime_dir)
        print(model_output_dir)
        # Check existence
        isExist = os.path.exists(model_output_dir)
        if not isExist:
            os.makedirs(model_output_dir)
        # Store the directory for this training process:
        self.current_output_dir = model_output_dir
        # Save data:
        model_to_save.save_pretrained(self.current_output_dir)
        self.tokenizer.save_pretrained(self.current_output_dir)

        label_map = {i: label for i, label in enumerate(label_list, 1)}
        # Save configurations:
        model_type = type(self.model).__name__
        model_config = {}
        try:
            model_config = self.model.model_config
        except Exception as e:
            print(e)
        model_config_extra = {"bert_model": self.train_config['--bert_model'],
                        "model_type": model_type,
                        "do_lower": self.train_config['--do_lower_case'],
                        "max_seq_length": self.train_config['--max_seq_length'],
                        "num_labels": len(label_list) + 1,
                        "label_map": label_map}

        json.dump(model_config, open(os.path.join(self.current_output_dir, "model_config.json"), "w"))
        json.dump(model_config_extra, open(os.path.join(self.current_output_dir, "model_config_extra.json"), "w"))
        json.dump(self.train_config, open(os.path.join(self.current_output_dir, "train_config.json"), "w"))

        # Start evaluating the model:
        self.model.to(device)

        if self.train_config['--is_test']:
            self.evaluate()

        return


    def evaluate(self, use_strict=True):
        # Get device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f'CUDA Available: {torch.cuda.is_available()}')
        print(device)
        # Set model to device:
        self.model.to(device)

        label_list = self.train_data_processors.get_labels(filter_toponym=self.train_config['--toponym_only'])


        eval_features = convert_examples_to_features(self.eval_examples, label_list,
                                                     self.train_config['--max_seq_length'], self.tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(self.eval_examples))
        logger.info("  Batch size = %d", self.train_config['--test_batch_size'])

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_valid_ids = torch.tensor([f.valid_ids for f in eval_features], dtype=torch.long)
        all_lmask_ids = torch.tensor([f.label_mask for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_valid_ids,
                                  all_lmask_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.train_config['--test_batch_size'])
        # Turn to eval mode:
        self.model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        label_map = {i: label for i, label in enumerate(label_list, 1)}
        for input_ids, input_mask, segment_ids, label_ids, valid_ids, l_mask in tqdm(eval_dataloader,
                                                                                     desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            valid_ids = valid_ids.to(device)
            label_ids = label_ids.to(device)
            l_mask = l_mask.to(device)

            with torch.no_grad():
                logits = self.model(input_ids, segment_ids, input_mask, valid_ids=valid_ids,
                                    attention_mask_label=l_mask)

            logits = torch.argmax(F.log_softmax(logits, dim=2), dim=2)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()

            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    if j == 0:
                        continue
                    elif label_ids[i][j] == len(label_map):
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j]])
                        temp_2.append(label_map[logits[i][j]])
        if use_strict:
            report = classification_report(y_true, y_pred, digits=4)
        else:
            y_true_loc_general = []
            y_pred_loc_general = []
            y_true_loc = []
            y_pred_loc = []
            print("Using Normal Evaluation Mode...")
            # Sanity check:
            if len(y_true) != len(y_pred):
                print("Evaluation data error!")
                report = None
            else:
                # Combining data:
                for i in range(len(y_true)):
                    if len(y_true[i]) != len(y_pred[i]):
                        print("Evaluation data mismatch, moving to the next data!")
                    else:
                        for item in y_true[i]:
                            if item == 'B-LOC' or item == 'I-LOC':
                                y_true_loc_general.append('LOC')
                                y_true_loc.append(item)
                            else:
                                y_true_loc_general.append('O')
                                y_true_loc.append('O')

                        for item in y_pred[i]:
                            if item == 'B-LOC' or item == 'I-LOC':
                                y_pred_loc_general.append('LOC')
                                y_pred_loc.append(item)
                            else:
                                y_pred_loc_general.append('O')
                                y_pred_loc.append('O')

            y_true_loc_general = np.array(y_true_loc_general)
            y_pred_loc_general = np.array(y_pred_loc_general)
            y_true_loc = np.array(y_true_loc)
            y_pred_loc = np.array(y_pred_loc)
            eval_label_list = ['B-LOC', 'I-LOC', 'O']
            eval_label_list_general = ['LOC', 'O']
            average_types = [None, 'micro', 'macro', 'weighted']
            report = ''
            # General location metrics:
            for avg_type in average_types:
                precision_general, recall_general, f1_general, support_general = precision_recall_fscore_support(
                    y_true_loc_general, y_pred_loc_general, labels=eval_label_list_general, average=avg_type)
                if avg_type is None:
                    for id, item in enumerate(eval_label_list_general):
                        report += 'Evaluation metrics for each class: \n'
                        report += f'{item} - PRF: {precision_general[id]} -- {recall_general[id]} -- {f1_general[id]} -- {support_general[id]} \n'
                else:
                    report += f'Evaluation metrics for {avg_type} average: \n'
                    report += f'{avg_type} - PRF: {precision_general} -- {recall_general} -- {f1_general} -- {support_general} \n'

            # Finer location metrics:
            for avg_type in average_types:
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true_loc, y_pred_loc, labels=eval_label_list, average=avg_type)
                if avg_type is None:
                    for id, item in enumerate(eval_label_list):
                        report += 'Evaluation metrics for each class: \n'
                        report += f'{item} - PRF: {precision[id]} -- {recall[id]} -- {f1[id]} -- {support[id]} \n'
                else:
                    report += f'Evaluation metrics for {avg_type} average: \n'
                    report += f'{avg_type} - PRF: {precision} -- {recall} -- {f1} -- {support} \n'

        if report is None:
            report = ""

        logger.info("\n%s", report)
        # Get report saving path:
        current_datetime = datetime.now()
        current_datetime_dir = current_datetime.strftime("%Y%m%d%H%M%S") + '\\'
        current_datetime_str = current_datetime.strftime("%Y%m%d%H%M%S")
        if self.current_output_dir is None:
            model_output_dir = os.path.join(self.train_config['--output_dir'], current_datetime_dir)
            print(model_output_dir)
            # Check existence
            isExist = os.path.exists(model_output_dir)
            if not isExist:
                os.makedirs(model_output_dir)
            # Store the directory for this training process:
            self.current_output_dir = model_output_dir
        eval_file_name = current_datetime_str + '_eval_results.txt'
        output_eval_file = os.path.join(self.current_output_dir, eval_file_name)
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            logger.info("\n%s", report)
            writer.write(report)
            writer.write("Test Processor Type: " + str(type(self.test_data_processors).__name__))
            writer.write("Use strict evaluation model: " + str(use_strict))
        return


# Main function, testing the codes:
def main():
    demo_model_args = {
        '--cuda': 'use GPU',
        '--pretrained_model': 'bert-base-cased',
        '--num_of_labels': 12,
        '--model_hidden_layer_size': 768,
        '--no_hidden_layers': 13,
        '--dropout': 0.1,
        '--out-channel': 16,
        '--verbose': 'whether to output the test results'
    }

    # Specify model:
    config = BertConfig.from_pretrained(demo_model_args['--pretrained_model'], num_labels=demo_model_args['--num_of_labels'], finetuning_task='geoparse')
    #model = BertSimpleNer.from_pretrained(demo_model_args['--pretrained_model'], from_tf=False, config=config)
    model = BertMlpNer(demo_model_args)

    current_trainer = TopoBertModelTrainer(model, _demo_train_config)
    current_trainer.train()


    return

if __name__ == '__main__':
    main()

