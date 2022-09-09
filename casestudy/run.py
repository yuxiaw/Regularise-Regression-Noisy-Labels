# coding=utf-8
# author: yuxia wang
# date: July 1, 2020
# update: 28 Dec, 2020 for ACL paper case study section experiments
"""BERT finetuning for STS task."""

from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import sys
import random
import pickle
import numpy as np
from tqdm import tqdm, trange
# from tqdm import tqdm_notebook as tqdm
# from tqdm import tnrange as trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, MSELoss, KLDivLoss

from pytorch_pretrained_bert.file_utils import WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertConfig, BertForSequenceClassification
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

# from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
from transformers import XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer
from transformers import XLMConfig, XLMForSequenceClassification, XLMTokenizer
from transformers import RobertaConfig,  RobertaForSequenceClassification,  RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer
from transformers import AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from tensorboardX import SummaryWriter

from args import parse_arguments, default_arguments
from run_classifier_dataset_utils import processors, output_modes, convert_examples_to_features, compute_metrics
from stsmodeling import HConvBertForSequenceClassification
# , BertForSequenceClassification   
from metrics import calculate_metrics

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    # ^^ safe to call this function even if cuda is not available
set_seed()
logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "hconvbert": (BertConfig, HConvBertForSequenceClassification, BertTokenizer),
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "auto": (AutoConfig, AutoModelForSequenceClassification, AutoTokenizer),
}

# save training arguments
def save_args(args, output_args_file):
    with open(output_args_file, 'w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s \n' % (key, value))
            # print(key, value) 

# load training arguments
def load_args(output_args_file):
    arguments = []
    with open(output_args_file) as file:
        lines = file.readlines()
        assert(len(lines) == 26)
    for line in lines:
        key, value = line.strip().split(":")
        arguments.append(value)
    args = default_arguments(arguments)
    args.task_name = args.task_name.lower()
    device = args.device
    args.device = torch.device("cpu") if device == "cpu" else torch.device("cuda")
    return args

def load_inference_examples(args, tokenizer, data_file):
    task_name = args.task_name.lower()
    output_mode = output_modes[task_name]
    processor = processors["stsinfer"]()
    label_list = processor.get_labels()

    # Prepare data loader
    examples = processor.get_test_examples(data_file)    
    logger.info("  Num examples = %d", len(examples))
    features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args.local_rank == -1:
        sampler = SequentialSampler(data)
        batch_size = args.eval_batch_size
    else:
        sampler = DistributedSampler(data)  # Note that this sampler samples randomly
    dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
    return dataloader

def load_and_cache_examples(args, tokenizer, evaluate=False):
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    label_list = processor.get_labels()
    output_mode = output_modes[task_name]
    prefix = "dev" if evaluate else "train"

    # Prepare data loader
    if evaluate: 
        logger.info("***** Running evaluation *****")
        examples = processor.get_dev_examples(args.data_dir)
    else:
        logger.info("***** Running Training *****")
        examples = processor.get_train_examples(args.data_dir) 
    logger.info("  Num examples = %d", len(examples))

    cached_features_file = os.path.join(args.data_dir, '{0}_{1}_{2}_{3}'.format(prefix,
        list(filter(None, args.weight_path_or_name.split('/'))).pop(),
                    str(args.max_seq_length),
                    str(task_name)))
    try:
        with open(cached_features_file, "rb") as reader:
            features = pickle.load(reader)
    except:
        features = convert_examples_to_features(
            examples, label_list, args.max_seq_length, tokenizer, output_mode)
        if args.local_rank == -1 or torch.distributed.get_rank() == 0:
            logger.info("  Saving %s features into cached file %s", prefix, cached_features_file)
            with open(cached_features_file, "wb") as writer:
                pickle.dump(features, writer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    if args.local_rank == -1:
        if evaluate:
            sampler = SequentialSampler(data)
            batch_size = args.eval_batch_size
        else:
            sampler = RandomSampler(data)
            batch_size = args.train_batch_size
    else:
        sampler = DistributedSampler(data)  # Note that this sampler samples randomly
    dataloader = DataLoader(data, sampler = sampler, batch_size = batch_size)
    # save vocab.txt
    if not os.path.exists(os.path.join(args.output_dir, "vocab.txt")):
        print("saving vocabulary!")
        tokenizer.save_vocabulary(args.output_dir)
    return dataloader


def train(args):
    # whether the output directory exists and is not empty
    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)
    
    # log loss and lr in tensorboardX
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # load toknizer, model
    task_name = args.task_name.lower()
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # config = config_class.from_pretrained(args.weight_path_or_name, num_labels=num_labels, finetuning_task=task_name)
    tokenizer = tokenizer_class.from_pretrained(args.weight_path_or_name, do_lower_case=args.do_lower_case)
    

    # Prepare Data
    train_dataloader = load_and_cache_examples(args, tokenizer, evaluate=False)
    eval_dataloader = load_and_cache_examples(args, tokenizer, evaluate=True)
    dataloaders = {
    "train": train_dataloader,
    "validation": eval_dataloader
    }
    
    # Load model
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    model = model_class.from_pretrained(args.weight_path_or_name, num_labels=num_labels)

    if args.local_rank == 0:
        torch.distributed.barrier()

    if args.fp16:
        model.half()
    model.to(args.device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    elif args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Set Loss function 
    # loss_ce = CrossEntropyLoss()
    # loss_mse = MSELoss()   # https://pytorch.org/docs/stable/nn.html#mseloss
    # loss_kld = KLDivLoss() # https://pytorch.org/docs/stable/nn.html#kldivloss 
    
    # Set optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                bias_correction=False,
                                max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                                t_total=num_train_optimization_steps)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                                lr=args.learning_rate,
                                warmup=args.warmup_proportion,
                                t_total=num_train_optimization_steps)

    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)                  

    # Update parameters
    global_step = 0
    r_best = 0.0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        """ In each epoch, first train, and then validate """
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                preds = []
            
            loss_epoch = 0
            for (batch_index, batch) in enumerate(tqdm(dataloaders[phase], desc = "Iteration")):
                batch = tuple(t.to(args.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                logits = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask)

                if output_mode == "classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
                elif output_mode == "regression":
                    loss_fct = MSELoss()
                    loss = loss_fct(logits.view(-1), label_ids.view(-1))
                    # loss = loss_mse(label_ids, logits.squeeze())  # loss = loss_kld(label_ids, y.squeeze())
                
                loss_epoch += loss.item() * label_ids.size(0)

                if phase == "train":
                    """ In training stage, parameters update """
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    global_step += 1

                    tb_writer.add_scalar('lr', optimizer.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', loss.item(), global_step)

                if phase == 'validation':
                    """ In Eval stage, collect predicted labels of all batches 
                    As there may be several groups of results that need to be evaluated
                    Correspinding to several outputs from models, like for regression and
                    classification at the same time, preds[0], preds[1] ... each is a track"""
                    if len(preds) == 0:
                        preds.append(logits.detach().cpu().numpy())
                    else:
                        preds[0] = np.append(preds[0], logits.detach().cpu().numpy(), axis = 0)

            # record in logger and draw figures for each epoch
            if phase == 'validation':
                ms = calculate_metrics(np.squeeze(preds[0]), 
                                dataloaders[phase].dataset.tensors[3].numpy(),
                                len(dataloaders["train"].dataset.tensors[3]))
                # save predictions
                preds = np.squeeze(preds[0])
                with open(os.path.join(args.data_dir, "train_preds_{}_epoch.txt".format(_)), "w") as file:
                    preds = preds.round(decimals=2)
                    file.write("\n".join([str(j) for j in preds]))
                # r = compute_metrics(task_name, np.squeeze(preds[0]), dataloaders[phase].dataset.tensors[3].numpy())
                # l = loss_epoch / len(dataloaders[phase].dataset)
                logger.info("***** Eval results of %s Epoch *****", str(_))
                print("Eval results of %d epoch" % _)
                # logger.info("%s, loss: %s", phase, str(l))
                # for key in sorted(r.keys()):
                #     print("%s = %s" % (key, str(r[key])))
                #     logger.info("%s = %s", key, str(r[key]))
                for key in sorted(ms.keys()):
                    print("%s = %s" % (key, str(ms[key])))
                    logger.info("%s = %s", key, str(ms[key]))
            else:
                l = loss_epoch / len(dataloaders[phase].dataset)
                logger.info("%s, loss: %s", phase, str(l))
        
        
        # Save a trained model, configuration
        print("Save a trained model and configuration")
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        # stop at the fifth epoch
        if _ == 4:
            break    
        # r_current = float(r["pearson"])
        # r_current = float(ms["dev_r"])
        # if r_current > r_best:
        #     r_best = r_current
        #     # Save a trained model, configuration
        #     print("Save a trained model and configuration")
        #     model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        #     # If we save using the predefined names, we can load using `from_pretrained`
        #     output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        #     output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        #     torch.save(model_to_save.state_dict(), output_model_file)
        #     model_to_save.config.to_json_file(output_config_file)

    output_args_file = os.path.join(args.output_dir, 'arguments.txt')
    save_args(args, output_args_file)
    tb_writer.close()


def evaluate(args):
    # Prepare data
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    dataloader = load_and_cache_examples(args, tokenizer, evaluate=True)

    # Load model
    model = model_class.from_pretrained(args.output_dir, num_labels=1)
    model.to(args.device)
    model.eval()

    # Predict
    preds = []
    for (batch_index, batch) in enumerate(tqdm(dataloader, desc = "Iteration")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, token_type_ids, label_ids = batch
        y = model(input_ids, token_type_ids, input_mask)
        if len(preds) == 0:
            preds.append(y.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], y.detach().cpu().numpy(), axis=0)
    preds = np.squeeze(preds[0])
    label_ids = dataloader.dataset.tensors[3].numpy()

    # Compute metric
    r = compute_metrics(args.task_name.lower(), preds, label_ids)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results *****")
        for key in sorted(r.keys()):
            logger.info("%s = %s", key, str(r[key]))
            writer.write("%s = %s\n" % (key, str(r[key])))
            print("{} = {}".format(key, str(r[key])))
            
    # save the predicted score in "preds.txt"
    with open(os.path.join(args.output_dir, "preds.txt"), "w") as file:
        preds = preds.round(decimals=2)
        file.write("\n".join([str(i) + "\t" + str(j) for i,j in zip(label_ids, preds)]))

def inference(args_file, data_file = None):
    # load arguments
    args = load_args(args_file)

    # Prepare data
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.output_dir if args.output_dir else args.weight_path_or_name, 
        do_lower_case=args.do_lower_case)
    
    if data_file is None:
        data_file = os.path.join(args.data_dir, "test.txt")
    dataloader = load_inference_examples(args, tokenizer, data_file)

    # Load model
    model = model_class.from_pretrained(
        args.output_dir if args.output_dir else args.weight_path_or_name,
        num_labels=1)
    model.to(args.device)
    model.eval()

    # Predict
    preds = []
    for (batch_index, batch) in enumerate(tqdm(dataloader, desc = "Iteration")):
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, input_mask, token_type_ids, label_ids = batch
        y = model(input_ids, token_type_ids, input_mask)
        if len(preds) == 0:
            preds.append(y.detach().cpu().numpy())
        else:
            preds[0] = np.append(preds[0], y.detach().cpu().numpy(), axis=0)
    preds = np.squeeze(preds[0])

    with open(data_file) as file:
        lines = file.readlines()     
    lines = [l.strip() for l in lines]   
    # save the predicted score and sentence pairs in "test_preds.txt"
    with open(os.path.join(args.output_dir, data_file[-7:-4]+"_preds.txt"), "w") as file:
        preds = preds.round(decimals=2)
        file.write("\n".join([i + "\t" + str(j) for i,j in zip(lines, preds)]))

def main():
    # load arguments
    args = parse_arguments(sys.argv)
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    args.device = device
    args.n_gpu = n_gpu

    # Setup logging
    logging.basicConfig(filename = args.logfile_dir,
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    set_seed(args.seed)

    # args setting is done!
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")
    if args.do_train:
        train(args)
    if args.do_eval:
        evaluate(args)
    
if __name__ == "__main__":
    main()
    