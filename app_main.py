#!/usr/bin/env python
# -*- encoding: utf-8 -*-
''' Document Informations:
@Project    :  PyCharm
@File       :  app_main.py
@Contact    :  bing_zhou_barrett@hotmail.com
@License    :  (C)Copyright 2021-2022, Bing Zhou, TAMU
@Author     :  Bing Zhou (Barrett)
@Modify Time:  2022/5/23 19:13
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
from bzrs_main.modules.ml_models.topo_bert.dataset_process import *
from bzrs_main.modules.ml_models.topo_bert.backbone_models import *
from bzrs_main.modules.ml_models.topo_bert.model_trainer import *

# Switches:


# Entrance
def main():
    # Configs:
    USE_WORKSTATION = False
    '''
        Experiment number:
        0=all; 1=
    '''
    SELECTED_EXP = [10, 12] # 0=all, 1

    bert_base_model_args = {
        "--cuda": "use GPU",
        "--pretrained_model": "bert-base-cased",
        "--num_of_labels": 12,
        "--model_hidden_layer_size": 768,
        "--no_hidden_layers": 13,
        "--dropout": 0.1,
        "--out-channel": 16,
        "--freeze-bert": False,
        "--verbose": "whether to output the test results"
    }

    bert_large_model_args = {
        "--cuda": "use GPU",
        "--pretrained_model": "bert-large-cased",
        "--num_of_labels": 12,
        "--model_hidden_layer_size": 1024,
        "--no_hidden_layers": 24,
        "--dropout": 0.1,
        "--out-channel": 16,
        "--freeze-bert": False,
        "--verbose": "whether to output the test results"
    }

    model_args_used = bert_base_model_args

    exp_train_config = {
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
        "--training_epoch": 1,
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

    # Get and combine dataset for only locations:
    conll_train_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\'
    conll_train_file = 'train.txt'
    conll_test_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\'
    conll_test_file = 'test.txt'
    wiki_train_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\Wikipedia3000\\'
    wiki_train_file = 'Wikipedia3000_feature_added.txt'
    wnut_train_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\WNUT2017\\'
    wnut_train_file = 'WNUT2017_feature_added.txt'
    harvey_test_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\HarveyTweet2017\\'

    if USE_WORKSTATION:
        model_args_used = bert_large_model_args
        exp_train_config = {
            "--task_name": "bert_geoparsing",
            "--toponym_only": False,
            "--random_seed": 42,
            "--use_gpu": 1,
            "--train_data_type": "conll",
            "--validate_data_type": "conll",
            "--test_data_type": "conll",
            "--train_data_dir": "D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
            "--validate_data_dir": "D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
            "--test_data_dir": "D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\",
            "--train_data_file": "train.txt",
            "--validate_data_file": "test.txt",
            "--test_data_file": "test.txt",
            "--is_validate": 1,
            "--is_test": 1,
            "--output_dir": "./outputs",
            "--cache_dir": "./cache",
            "--bert_model": "bert-large-cased",
            "--do_lower_case": False,
            "--max_seq_length": 128,
            "--training_epoch": 50,
            "--train_batch_size": 32,
            "--test_batch_size": 32,
            "--learning_rate": 5e-5,
            "--warm_up_proportion": 0.1,
            "--weight_decay": 0.01,
            "--adam_epsilon": 1e-8,
            "--max_grad_norm": 1.0,
            "--num_grad_accum_steps": 1,
            "--loss_scale": 0
        }

        # Get and combine dataset for only locations:
        conll_train_dir = 'D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\'
        conll_train_file = 'train.txt'
        conll_test_dir = 'D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\'
        conll_test_file = 'test.txt'
        wiki_train_dir = 'D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\Wikipedia3000\\'
        wiki_train_file = 'Wikipedia3000_feature_added.txt'
        wnut_train_dir = 'D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\WNUT2017\\'
        wnut_train_file = 'WNUT2017_feature_added.txt'
        harvey_test_dir = 'D:\\BarrettExclusiveSpace\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\datasets\\HarveyTweet2017\\'

    # Prepare customized datasets:
    data_processors_conll = Processor_CoNLL2003()
    data_processors_standard = Processor_Standard_V1()
    data_processors_harvey = Processor_Harvey()

    total_train_examples_loc = []
    total_train_examples_ner = []

    conll_train_example_loc = data_processors_conll.get_train_data(conll_train_dir, conll_train_file, True)
    wiki_train_example_loc = data_processors_standard.get_train_data(wiki_train_dir, wiki_train_file, True)
    wnut_train_example_loc = data_processors_standard.get_train_data(wnut_train_dir, wnut_train_file, True)

    conll_train_example_ner = data_processors_conll.get_train_data(conll_train_dir, conll_train_file)
    wiki_train_example_ner = data_processors_standard.get_train_data(wiki_train_dir, wiki_train_file)
    wnut_train_example_ner = data_processors_standard.get_train_data(wnut_train_dir, wnut_train_file)

    conll_test_example_loc = data_processors_conll.get_test_data(conll_test_dir, conll_test_file, True)
    conll_test_example_ner = data_processors_conll.get_test_data(conll_test_dir, conll_test_file)
    harvey_test_example = data_processors_harvey.get_test_data(harvey_test_dir)

    total_train_examples_loc.extend(conll_train_example_loc)
    total_train_examples_loc.extend(wiki_train_example_loc)
    total_train_examples_loc.extend(wnut_train_example_loc)

    total_train_examples_ner.extend(conll_train_example_ner)
    total_train_examples_ner.extend(wiki_train_example_ner)
    total_train_examples_ner.extend(wnut_train_example_ner)

    # Default model config:
    config = BertConfig.from_pretrained(exp_train_config['--bert_model'],
                                        num_labels=model_args_used['--num_of_labels'], finetuning_task='geoparse')

    # **************************   Experiment Starts:   ********************************
    # 1: conll + bert_large(base) + linear / conll + harvey / ner
    if 0 in SELECTED_EXP or 1 in SELECTED_EXP:
        print("conll + bert_large(base) + linear / conll + harvey / ner")
        model = BertSimpleNer.from_pretrained(model_args_used['--pretrained_model'], from_tf=False, config=config)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 2: conll + bert_large(base) + mlp / conll + harvey / ner
    if 0 in SELECTED_EXP or 2 in SELECTED_EXP:
        print("conll + bert_large(base) + mlp / conll + harvey / ner")
        model = BertMlpNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 3: conll + bert_large(base) + 1dCNN / conll + harvey / ner
    if 0 in SELECTED_EXP or 3 in SELECTED_EXP:
        print("conll + bert_large(base) + 1dCNN / conll + harvey / ner")
        model = BertCNN1DNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 4: conll&wiki&wnut + bert_large(base) + linear / conll + harvey / ner
    if 0 in SELECTED_EXP or 4 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + linear / conll + harvey / ner")
        model = BertSimpleNer.from_pretrained(model_args_used['--pretrained_model'], from_tf=False, config=config)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.customize_datasets(total_train_examples_ner, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_ner
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()

    # 5: conll&wiki&wnut + bert_large(base) + mlp / conll + harvey / ner
    if 0 in SELECTED_EXP or 5 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + mlp / conll + harvey / ner")
        model = BertMlpNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config=exp_train_config)
        current_trainer.customize_datasets(total_train_examples_ner, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_ner
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()

    # 6: conll&wiki&wnut + bert_large(base) + 1dCNN / conll + harvey / ner
    if 0 in SELECTED_EXP or 6 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + 1dCNN / conll + harvey / ner")
        model = BertCNN1DNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config=exp_train_config)
        current_trainer.customize_datasets(total_train_examples_ner, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_ner
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()

    # 7: conll + bert_large(base) + linear / conll + harvey / loc
    if 0 in SELECTED_EXP or 7 in SELECTED_EXP:
        print("conll + bert_large(base) + linear / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertSimpleNer.from_pretrained(model_args_used['--pretrained_model'], from_tf=False, config=config)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 8: conll + bert_large(base) + mlp / conll + harvey / loc
    if 0 in SELECTED_EXP or 8 in SELECTED_EXP:
        print("conll + bert_large(base) + mlp / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertMlpNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 9: conll + bert_large(base) + 1dcnn / conll + harvey / loc
    if 0 in SELECTED_EXP or 9 in SELECTED_EXP:
        print("conll + bert_large(base) + 1dcnn / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertCNN1DNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config = exp_train_config)
        current_trainer.train()
        current_trainer.eval_examples = harvey_test_example
        current_trainer.test_data_processors = data_processors_harvey
        current_trainer.evaluate()

    # 10: conll&wiki&wnut + bert_large(base) + linear / conll + harvey / loc
    if 0 in SELECTED_EXP or 10 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + linear / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertSimpleNer.from_pretrained(model_args_used['--pretrained_model'], from_tf=False, config=config)
        current_trainer = TopoBertModelTrainer(model, train_config=exp_train_config)
        current_trainer.customize_datasets(total_train_examples_loc, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_loc
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()

    # 11: conll&wiki&wnut + bert_large(base) + mlp / conll + harvey / loc
    if 0 in SELECTED_EXP or 11 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + mlp / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertMlpNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config=exp_train_config)
        current_trainer.customize_datasets(total_train_examples_loc, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_loc
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()

    # 12: conll&wiki&wnut + bert_large(base) + 1dCNN / conll + harvey / loc
    if 0 in SELECTED_EXP or 12 in SELECTED_EXP:
        print("conll&wiki&wnut + bert_large(base) + 1dCNN / conll + harvey / loc")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model = BertCNN1DNer(model_config=model_args_used)
        current_trainer = TopoBertModelTrainer(model, train_config=exp_train_config)
        current_trainer.customize_datasets(total_train_examples_loc, harvey_test_example, None, data_processors_harvey)
        current_trainer.train()
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_loc
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate()


    # Evaluate using exsiting model:
    if 0 in SELECTED_EXP or 13 in SELECTED_EXP:
        print("Evaluate using exsiting model:")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\outputs\\20220529205830\\'
        model_dir = 'D:\\BarrettExclusiveSpace\\GeoparseEval\\20220526135822_in_use\\'
        # Get model config:
        current_model_config = os.path.join(model_dir, 'model_config.json')
        current_model_config = json.load(open(current_model_config))
        # Get training config:
        current_training_config = os.path.join(model_dir, 'train_config.json')
        current_training_config = json.load(open(current_training_config))
        # Setup model and trainer:
        #model = BertCNN1DNer.from_pretrained(model_dir, model_config=current_model_config)
        model = BertMlpNer.from_pretrained(model_dir, model_config=current_model_config)
        current_trainer = TopoBertModelTrainer(model, train_config=current_training_config)

        current_trainer.customize_datasets(total_train_examples_loc, harvey_test_example, None, data_processors_harvey)
        # Use the same output dir:
        current_trainer.current_output_dir = model_dir
        current_trainer.evaluate(use_strict=False)
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_loc
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate(use_strict=False)

    # Load existing model for Simple Model:
    if 0 in SELECTED_EXP or 14 in SELECTED_EXP:
        print("Evaluate using exsiting model:")
        model_args_used['--num_of_labels'] = 6
        exp_train_config['--toponym_only'] = True
        model_dir = 'E:\\CodingProjects\\GRITFramework\\BZResearchStack\\src\\bzrs_main\\modules\\ml_models\\topo_bert\\outputs\\20220529205830\\'
        model_dir = 'D:\\BarrettExclusiveSpace\\GeoparseEval\\20220526135822_in_use\\'
        # Get model config:
        current_model_config = os.path.join(model_dir, 'model_config.json')
        current_model_config = json.load(open(current_model_config))
        # Get training config:
        current_training_config = os.path.join(model_dir, 'train_config.json')
        current_training_config = json.load(open(current_training_config))

        model = BertSimpleNer.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir, do_lower_case=current_training_config['--do_lower_case'])
        current_trainer = TopoBertModelTrainer(model, train_config=current_training_config)

        current_trainer.customize_datasets(total_train_examples_loc, harvey_test_example, None, data_processors_harvey)
        # Use the same output dir:
        current_trainer.current_output_dir = model_dir
        current_trainer.evaluate(use_strict=False)
        # Another evaluation:
        current_trainer.eval_examples = conll_test_example_loc
        current_trainer.test_data_processors = data_processors_conll
        current_trainer.evaluate(use_strict=False)

    return

if __name__ == '__main__':
    main()