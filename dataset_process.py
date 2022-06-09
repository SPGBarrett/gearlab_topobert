#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  dataset_process.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2022/5/23 19:12
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
import os
import logging
import string
from os import listdir
from os.path import isfile, join

logger = logging.getLogger(__name__)

class InputTextAndLabels(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


class DataProcessor(object):
    """Super class for data converters for sequence classification data sets."""

    def get_train_data(self, data_dir, file_name, filter_toponym=False):
        return self._create_examples(
                    self._read_tsv(os.path.join(data_dir, file_name), filter_toponym=filter_toponym), "train")

    def get_dev_data(self, data_dir, file_name, filter_toponym=False):
        """Gets a collection of `InputExample`s for the dev set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name), filter_toponym=filter_toponym), "dev")

    def get_test_data(self, data_dir, file_name, filter_toponym=False):
        """Gets a collection of `InputExample`s for the test set."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name), filter_toponym=filter_toponym), "test")

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        raise NotImplementedError()

    def _read_tsv(self, input_file, quotechar=None, filter_toponym=False):
        """Reads a tab separated value file."""
        raise NotImplementedError()

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputTextAndLabels(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class Processor_CoNLL2003(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_labels(self, filter_toponym=False):
        if filter_toponym == False:
            return ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]
        else:
            return ["O", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _read_tsv(self, input_file, quotechar=None, filter_toponym=False):
        """Reads a tab separated value file."""
        data = []
        sentence = []
        label = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                splits = line.split(' ')
                sentence.append(splits[0])
                current_label = splits[-1][:-1]
                if filter_toponym == False:
                    label.append(current_label)
                else:
                    if current_label == 'B-LOC' or current_label == 'I-LOC':
                        label.append(current_label)
                    else:
                        label.append('O')

            # For the last line:
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []

        return data



class Processor_Standard_V1(DataProcessor):
    """Processor for the Wiki3000 and WNUT2017 dataset."""

    def get_labels(self):
        return ["O", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _read_tsv(self, input_file, quotechar=None, filter_toponym=False):
        """Reads a tab separated value file."""
        data = []
        sentence = []
        label = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        data.append((sentence, label))
                        sentence = []
                        label = []
                    continue
                splits = line.split('\t')
                sentence.append(splits[0].strip())
                if len(splits) > 1:
                    current_label = splits[1].strip()
                    if current_label == 'B-location' or current_label.startswith('B-'):
                        label.append('B-LOC')
                    elif current_label == 'I-location' or current_label.startswith('I-'):
                        label.append('I-LOC')
                    else:
                        label.append('O')

                else:
                    label.append('O')
                # label.append(splits[-1][:-1])
            # For the last line:
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []

        return data




class Processor_Harvey(DataProcessor):
    """Processor for the Harvey Tweets dataset."""
    def get_train_data(self, data_dir):
        """See base class."""
        return self._create_examples(self.combine_tsv_data(data_dir), "train")

    def get_dev_data(self, data_dir):
        """See base class."""
        return self._create_examples(self.combine_tsv_data(data_dir), "dev")

    def get_test_data(self, data_dir):
        """See base class."""
        return self._create_examples(self.combine_tsv_data(data_dir), "test")

    def get_labels(self):
        return ["O", "B-LOC", "I-LOC", "[CLS]", "[SEP]"]

    def _read_tsv(self, input_file, quotechar=None, filter_toponym=False):
        """Reads a tab separated value file."""
        sentence = []
        label = []
        data = None
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
                    if len(sentence) > 0:
                        data = (sentence, label)
                        sentence = []
                        label = []
                    continue
                splits = line.split('\t')
                sentence.append(splits[0].strip())
                if len(splits) > 1:
                    current_label = splits[1].strip()
                    if current_label == 'B-location' or current_label.startswith('B-'):
                        label.append('B-LOC')
                    elif current_label == 'I-location' or current_label.startswith('I-'):
                        label.append('I-LOC')
                    else:
                        label.append('O')
                else:
                    label.append('O')
            # For the last line:
            if len(sentence) > 0:
                data = (sentence, label)
                sentence = []
                label = []

        return data

    def combine_tsv_data(self, data_dir):
        data_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        data_list = []
        for file in data_files:
            full_dir = join(data_dir, file)
            data_content = self._read_tsv(full_dir)
            data_list.append(data_content)
        return data_list


    def read_sentence_and_label(self, data_dir):
        sentence_list = []
        sentence_label_list = []
        punctuation_list = list(string.punctuation)
        data_files = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
        for file in data_files:
            full_dir = join(data_dir, file)
            with open(full_dir, 'r', encoding='utf-8') as f:
                tmp_string = ''
                label_triple = None
                label_triple_list = []
                for line in f:
                    line_preprocess = str(line).strip().replace('\n', '')
                    if line_preprocess in punctuation_list:
                        continue
                    line_content = str(line_preprocess).split('\t')
                    if len(line_content) > 1:  # Label exist
                        str_content = line_content[0]
                        line_label = line_content[1]
                        label_triple = (str_content, line_label)
                    else:  # No label
                        str_content = line_content[0]
                        label_triple = (str_content, 0)

                    label_triple_list.append(label_triple)
                    tmp_string += (' ' + str_content)
                tmp_string = tmp_string.strip()
                print(tmp_string)
                print(label_triple_list)
                sentence_label_list.append(label_triple_list)
                sentence_list.append(tmp_string)

        # Print final result:
        print(sentence_list)
        print(sentence_label_list)
        return (sentence_list, sentence_label_list)

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}

    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        if len(textlist) != len(labellist):
            print("Current data has issues:")
            print(str(textlist))
            print(str(labellist))
            continue
        tokens = []
        labels = []
        valid = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid.append(1)
                    label_mask.append(1)
                else:
                    valid.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid = valid[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid.insert(0,1)
        label_mask.insert(0,1)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid.append(1)
        label_mask.append(1)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [1] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid.append(1)
            label_mask.append(0)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid) == max_seq_length
        assert len(label_mask) == max_seq_length

        # Display converted data samples for the first 5 entries of data:
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            # logger.info("label: %s (id = %d)" % (example.label, label_ids))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids,
                              valid_ids=valid,
                              label_mask=label_mask))
    return features


def print_examples(data_example):
    for index, current_data in enumerate(data_example):
        print(current_data.text_a)
        print(current_data.label)
        if index > 4:
            break
    return



def main():
    # Testing functions:
    CONLL2003_DATA_DIR = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\'
    CONLL2003_DATA_FILE = 'train.txt'
    WIKI3000_DIR = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\Wikipedia3000\\'
    WIKI3000_FILE = 'Wikipedia3000_feature_added.txt'
    WNUT2017_DIR = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\WNUT2017\\'
    WNUT2017_FILE = 'WNUT2017_feature_added.txt'
    HARVEY_TWEET_DIR = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\HarveyTweet2017\\'

    # Getting original data:
    conll_processor = Processor_CoNLL2003()
    standard_processor = Processor_Standard_V1()
    harvey_processor = Processor_Harvey()

    conll_data = conll_processor.get_train_data(CONLL2003_DATA_DIR, CONLL2003_DATA_FILE, False)
    wiki_data = standard_processor.get_train_data(WIKI3000_DIR, WIKI3000_FILE)
    wunt_data = standard_processor.get_train_data(WNUT2017_DIR, WNUT2017_FILE)
    harvey_data = harvey_processor.get_train_data(HARVEY_TWEET_DIR)

    print_examples(conll_data)
    print_examples(wiki_data)
    print_examples(wunt_data)
    print_examples(harvey_data)

    # Getting only LOC data:
    conll_data_loc = conll_processor.get_train_data(CONLL2003_DATA_DIR, CONLL2003_DATA_FILE, True)
    print_examples(conll_data_loc)

    return

# Main function for testing
if __name__ == '__main__':
    main()