#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  topo_bert.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2022/5/23 19:12
@Version    :  1.0
@Description:  The application interfaces.

Example:
    Some examples of usage here.
Attributes:
    Attribute description here.
Todo:
    * For module TODOs

'''

# import lib
import nltk
from bzrs_main.modules.ml_models.topo_bert.backbone_models import *
from bzrs_main.modules.ml_models.topo_bert.dataset_process import *
import os
import json
import re
from nltk import word_tokenize
import queue


DEFAULT_TOPOBERT_PATH = './pretrained_models/topobert_cnn1d/'
nltk.download('punkt')

class TopoBERT:

    def __init__(self, model_dir: str = DEFAULT_TOPOBERT_PATH):
        '''

        Args:
            model_dir (): Locate the dir that stores all the model files, or simply use the default path.
        '''
        try:
            self.model , self.tokenizer, self.model_config, self.training_config = self.load_model(model_dir)
            self.label_map = {"1": "O", "2": "B-LOC", "3": "I-LOC", "4": "[CLS]", "5": "[SEP]"}
            self.max_seq_length = self.training_config["--max_seq_length"]
            self.label_map = {int(k): v for k, v in self.label_map.items()}
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Add model to device and set eval mode:
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(e)

    def load_model(self, model_dir: str, model_config: str = "model_config.json"):
        # Load model config:
        model_config_file = os.path.join(model_dir, model_config)
        current_model_config = json.load(open(model_config_file))
        # Init model and load pretrained params:
        #model = BertSimpleNer.from_pretrained(model_dir)
        model = BertCNN1DNer.from_pretrained(model_dir, model_config=current_model_config)
        # Get training config:
        current_training_config = os.path.join(model_dir, 'train_config.json')
        current_training_config = json.load(open(current_training_config))
        # Init tokenizer:
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=current_training_config['--do_lower_case'])

        # Return all params:
        return model, tokenizer, current_model_config, current_training_config


    def tokenize(self, text: str):
        """ tokenize input"""
        words = word_tokenize(text)
        tokens = []
        valid_positions = []
        for i, word in enumerate(words):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for i in range(len(token)):
                if i == 0:
                    valid_positions.append(1)
                else:
                    valid_positions.append(0)
        return tokens, valid_positions


    def preprocess(self, text: str):
        """ preprocess """
        tokens, valid_positions = self.tokenize(text)
        ## insert "[CLS]"
        tokens.insert(0, "[CLS]")
        valid_positions.insert(0, 1)
        ## insert "[SEP]"
        tokens.append("[SEP]")
        valid_positions.append(1)
        segment_ids = []
        for i in range(len(tokens)):
            segment_ids.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length: # Padding
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            valid_positions.append(0)
        return input_ids,input_mask,segment_ids,valid_positions


    def prettify_result(self, org_result):
        '''
        Input the original result list from the TopoBERT and formulate the
        Args:
            org_result (str): list of the predicted result for each token.

        Returns:
            Prettified result output with extra information
        '''
        combined_addresses = []  # A list of all address
        address_results = []  # A list of all predicted LOC
        full_address = ''  # Link all addresses in combined_addresses
        tmp_queue = queue.Queue(maxsize=20)
        tmp_address = ''
        for index, content in enumerate(org_result):
            if content['tag'] == 'B-LOC':  # If B-LOC, clear, save and enqueue
                # If not empty, empty it and save data:
                if not tmp_queue.empty():
                    # Get all content out first:
                    while not tmp_queue.empty():
                        tmp_address += str(tmp_queue.get()) + ' '
                    tmp_address = tmp_address.strip()
                    combined_addresses.append(tmp_address)
                    tmp_address = ''
                # Enqueue
                tmp_queue.put(content['word'].strip())
                # Save location entity:
                address_results.append(content['word'].strip())
            elif content['tag'] == 'I-LOC':  # If I-LOC, enqueue directly
                # Enqueue
                tmp_queue.put(content['word'].strip())
                # Save location entity:
                address_results.append(content['word'].strip())
            else:  # Else, clear and save
                if not tmp_queue.empty():
                    # Get all content out first:
                    while not tmp_queue.empty():
                        tmp_address += str(tmp_queue.get()) + ' '
                    tmp_address = tmp_address.strip()
                    combined_addresses.append(tmp_address)
                    tmp_address = ''
        # Deal with remaining data:
        if not tmp_queue.empty():
            # Get all content out first:
            while not tmp_queue.empty():
                tmp_address += str(tmp_queue.get()) + ' '
            tmp_address = tmp_address.strip()
            combined_addresses.append(tmp_address)
            tmp_address = ''

        # Get Full address:
        for add_content in combined_addresses:
            full_address += ' ' + add_content
        full_address = full_address.strip()

        # Construct output result:
        result_dict = {
            'combined_addresses': combined_addresses,
            'full_address': full_address,
            'address_results': address_results
        }

        return result_dict


    def predict(self, text: str):
        input_ids, input_mask, segment_ids, valid_ids = self.preprocess(text)
        input_ids = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        input_mask = torch.tensor([input_mask], dtype=torch.long, device=self.device)
        segment_ids = torch.tensor([segment_ids], dtype=torch.long, device=self.device)
        valid_ids = torch.tensor([valid_ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(input_ids, segment_ids, input_mask, None, valid_ids, None)
        logits = F.softmax(logits,dim=2)
        logits_label = torch.argmax(logits,dim=2)
        logits_label = logits_label.detach().cpu().numpy().tolist()[0]

        logits_confidence = [values[label].item() for values, label in zip(logits[0], logits_label)]

        logits = []

        pos = 0
        for index, mask in enumerate(valid_ids[0]):
            if index == 0:
                continue
            if mask == 1:
                logits.append((logits_label[index-pos], logits_confidence[index-pos]))
            else:
                pos += 1
        logits.pop()

        labels = [(self.label_map[label], confidence) for label, confidence in logits]
        words = word_tokenize(text)
        assert len(labels) == len(words)
        output = [{"word":word, "tag":label, "confidence":confidence} for word, (label, confidence) in zip(words, labels)]
        prettified_result = self.prettify_result(output)
        result_dict = prettified_result
        result_dict['org_result'] = output

        return result_dict

    def extract_zip_from_text(self, text_data):
        us_zip_reg = r'(\d{5}\-?\d{0,4})'
        zip_code = re.search(us_zip_reg, text_data)
        if zip_code is not None and zip_code != "":
            result = zip_code.group(1)
        else:
            result = None
        return result

    def predict_with_ruless(self, text: str):
        '''
            Adding zipcode extractor to it
        Args:
            text ():

        Returns: identified location names with zip codes.

        '''
        zipcode_extracted = self.extract_zip_from_text(text)
        topo_bert_result = self.predict(text)
        # Combined output:
        result = {
            'topo_bert_result': topo_bert_result,
            'zipcode': zipcode_extracted
        }
        return result



def main():
    # Testing the functions:
    test_text = """HarveyStorm over Austin TX at 8: 00 AM CDT via Weather Underground"""
    # test_text = '''Houston HoustonFlood the intersection of I-45N. Main Street'''
    # model_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\outputs\\20220529205830\\'
    model_dir = 'D:\\BarrettExclusiveSpace\\GeoparseEval\\20220601200642_in_use_correct\\'

    current_geoparser = TopoBERT(model_dir)
    result = current_geoparser.predict(test_text)
    print(result)
    return 'End of Test...'


if __name__ == '__main__':
    main()