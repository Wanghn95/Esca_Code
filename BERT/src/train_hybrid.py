#!/usr/bin/env python
"""
    Main training workflow
"""
from __future__ import division

import argparse
import glob
import os
import random
import signal
import time
import torch
import distributed
from pytorch_transformers import BertTokenizer
from models import data_loader, model_builder
from models.data_loader import load_dataset
from models.model_builder import ExtSummarizer, HybridSummarizer
from models.loss import abs_loss
from models.trainer import build_trainer
from models.predictor_copy import build_predictor
from others.logging import logger, init_logger
model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']
# symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
#            'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}

def train_multi_hybrid(args):
    """ Spawns 1 process per GPU """
    # Each GPU generates one process.
    init_logger()
    nb_gpu = args.world_size
    # Do multithreading with the spawn method.
    mp = torch.multiprocessing.get_context('spawn')

    # Create a thread to listen for errors in the child processes.
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Train with multiprocessing.
    procs = []
    for i in range(nb_gpu):
        device_id = i
        # Add a new thread, this thread runs the run function, device_id is the corresponding GPU.
        procs.append(mp.Process(target=run, args=(args,
                                                  device_id, error_queue,), daemon=True))
        procs[i].start()
        logger.info(" Starting process pid: %d  " % procs[i].pid)
        #  Let the error handler monitor the child threads.
        error_handler.add_child(procs[i].pid)
    for p in procs:
        p.join()


def run(args, device_id, error_queue):
    """ run process """
    setattr(args, 'gpu_ranks', [int(i) for i in args.gpu_ranks])

    try:
        gpu_rank = distributed.multi_init(device_id, args.world_size, args.gpu_ranks)
        print('gpu_rank %d' % gpu_rank)
        if gpu_rank != args.gpu_ranks[device_id]:
            raise AssertionError("An error occurred in \
                  Distributed initialization")

        train_single_hybrid(args, device_id)
    except KeyboardInterrupt:
        pass  # killed by parent, do nothing
    except Exception:
        # propagate exception to parent process, keeping original traceback
        import traceback
        error_queue.put((args.gpu_ranks[device_id], traceback.format_exc()))


class ErrorHandler(object):
    """A class that listens for exceptions in children processes and propagates
    the tracebacks to the parent process."""

    def __init__(self, error_queue):
        """ init error handler """
        import signal
        import threading
        self.error_queue = error_queue
        self.children_pids = []
        self.error_thread = threading.Thread(
            target=self.error_listener, daemon=True)
        self.error_thread.start()
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """ error handler """
        self.children_pids.append(pid)

    def error_listener(self):
        """ error listener """
        (rank, original_trace) = self.error_queue.get()
        self.error_queue.put((rank, original_trace))
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, signalnum, stackframe):
        """ signal handler """
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)  # kill children processes
        (rank, original_trace) = self.error_queue.get()
        msg = """\n\n-- Tracebacks above this line can probably
                 be ignored --\n\n"""
        msg += original_trace
        raise Exception(msg)


def validate_hybrid(args, device_id):
    timestep = 0
    if (args.test_all):
        cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
        cp_files.sort(key=os.path.getmtime)
        xent_lst = []
        for i, cp in enumerate(cp_files):
            step = int(cp.split('.')[-2].split('_')[-1])
            xent = validate(args, device_id, cp, step)
            xent_lst.append((xent, cp))
            max_step = xent_lst.index(min(xent_lst))
            if (i - max_step > 10):
                break
        xent_lst = sorted(xent_lst, key=lambda x: x[0])[:3]
        logger.info('PPL %s' % str(xent_lst))
        for xent, cp in xent_lst:
            step = int(cp.split('.')[-2].split('_')[-1])
            test_hybrid(args, device_id, cp, step)
    else:
        while (True):
            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (not os.path.getsize(cp) > 0):
                    time.sleep(60)
                    continue
                if (time_of_cp > timestep):
                    timestep = time_of_cp
                    step = int(cp.split('.')[-2].split('_')[-1])
                    validate(args, device_id, cp, step)
                    test_hybrid(args, device_id, cp, step)

            cp_files = sorted(glob.glob(os.path.join(args.model_path, 'model_step_*.pt')))
            cp_files.sort(key=os.path.getmtime)
            if (cp_files):
                cp = cp_files[-1]
                time_of_cp = os.path.getmtime(cp)
                if (time_of_cp > timestep):
                    continue
            else:
                time.sleep(300)


def validate(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = HybridSummarizer(args, device, checkpoint)
    model.eval()

    valid_iter = data_loader.Dataloader(args, load_dataset(args, 'valid', shuffle=False),
                                        args.batch_size, device,
                                        shuffle=False, is_test=False)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    valid_loss = abs_loss(model.abstractor.generator, symbols, model.abstractor.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)

    trainer = build_trainer(args, device_id, model, None, valid_loss)
    stats = trainer.validate(valid_iter, step)
    return stats.xent()


def test_hybrid(args, device_id, pt, step):
    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    if (pt != ''):
        test_from = pt
    else:
        test_from = args.test_from
    logger.info('Loading checkpoint from %s' % test_from)
    checkpoint = torch.load(test_from, map_location=lambda storage, loc: storage)
    opt = vars(checkpoint['opt'])
    for k in opt.keys():
        if (k in model_flags):
            setattr(args, k, opt[k])
    print(args)

    model = HybridSummarizer(args, device, checkpoint)
    model.eval()

    test_iter = data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False),
                                       args.test_batch_size, device,
                                       shuffle=False, is_test=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    predictor = build_predictor(args, tokenizer, symbols, model, logger)
    predictor.translate(test_iter, step)

def train_hybrid(args, device_id):
    # Whether to choose multiple Gpus.
    if (args.world_size > 1):
        train_multi_hybrid(args)
    else:
        # print("~~~~~~~~~~~~~~~")
        train_single_hybrid(args, device_id)


def train_single_hybrid(args, device_id):
    init_logger(args.log_file)

    device = "cpu" if args.visible_gpus == '-1' else "cuda"
    logger.info('Device ID %d' % device_id)
    logger.info('Device %s' % device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from != '':
        logger.info('Loading checkpoint from %s' % args.train_from)
        checkpoint = torch.load(args.train_from,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint = None


    if args.train_from_extractor != '':
        logger.info('Loading checkpoint from %s' % args.train_from_extractor)
        checkpoint_ext = torch.load(args.train_from_extractor,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint_ext['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint_ext = None

    if args.train_from_abstractor != '':
        logger.info('Loading checkpoint from %s' % args.train_from_abstractor)
        checkpoint_abs = torch.load(args.train_from_abstractor,
                                map_location=lambda storage, loc: storage)
        opt = vars(checkpoint_abs['opt'])
        for k in opt.keys():
            if (k in model_flags):
                setattr(args, k, opt[k])
    else:
        checkpoint_abs = None

    def train_iter_fct():
        if args.is_debugging:
            print("YES it is debugging")
            return data_loader.Dataloader(args, load_dataset(args, 'test', shuffle=False), args.batch_size, device,
                                      shuffle=False, is_test=False)
            # exit()
        else:
            return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, device,
                                      shuffle=True, is_test=False)

    model = HybridSummarizer(args, device,  checkpoint, checkpoint_ext=checkpoint_ext, checkpoint_abs=checkpoint_abs)

    if (args.sep_optim):
        optim_bert = model_builder.build_optim_bert(args, model, checkpoint)
        optim_dec = model_builder.build_optim_dec(args, model, checkpoint)
        optim = [optim_bert, optim_dec]
    else:
        optim = [model_builder.build_optim(args, model, checkpoint)]

    logger.info(model)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir=args.temp_dir)
    symbols = {'BOS': tokenizer.vocab['[unused0]'], 'EOS': tokenizer.vocab['[unused1]'],
               'PAD': tokenizer.vocab['[PAD]'], 'EOQ': tokenizer.vocab['[unused2]']}
    train_loss = abs_loss(model.abstractor.generator, symbols, model.abstractor.vocab_size, device, train=True,
                          label_smoothing=args.label_smoothing)
    trainer = build_trainer(args, device_id, model, optim, train_loss)

    trainer.train(train_iter_fct, args.train_steps)
