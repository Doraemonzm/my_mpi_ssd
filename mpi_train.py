import argparse
import logging
import os


import math
import torch
import torch.distributed as dist

from ssd.data import samplers

from ssd.engine.inference import do_evaluation
from ssd.config import cfg
from ssd.data.build import make_data_loader
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch.utils.data.dataloader import default_collate
from ssd.utils.metric_logger import MetricLogger
from ssd.engine.trainer import do_train
from ssd.engine.inference import compute_on_dataset
from ssd.modeling.detector import build_detection_model
from ssd.solver.build import make_optimizer, make_lr_scheduler
from ssd.utils import dist_util, mkdir
from ssd.utils.checkpoint import CheckPointer
from ssd.utils.dist_util import synchronize
from ssd.utils.logger import setup_logger
import torch.nn as nn
from ssd.utils.misc import str2bool
from tqdm import tqdm
import numpy as np
from ssd.data.transforms import build_transforms, build_target_transform
from ssd.data.datasets import build_dataset
from ssd.structures.container import Container
from torch.utils.data.sampler import Sampler
from collections import OrderedDict
from ssd.data.datasets.evaluation import evaluate
from mpi4py import MPI
import time





class BatchCollator:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = default_collate(transposed_batch[0])
        img_ids = default_collate(transposed_batch[2])
        # print('len(transposed_batch)', len(transposed_batch))
        # print('transposed_batch[0]', transposed_batch[0])
        # print('transposed_batch[1]', transposed_batch[1])
        # print('transposed_batch[2]', transposed_batch[2])
        if self.is_train:
            list_targets = transposed_batch[1]
            targets = Container(
                {key: default_collate([d[key] for d in list_targets]) for key in list_targets[0]}
            )
        else:
            targets = None
        return images, targets, img_ids



def my_data_loader(cfg, rank, size, is_train=True):
    train_transform = build_transforms(cfg, is_train=is_train)
    target_transform = build_target_transform(cfg) if is_train else None
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    datasets = build_dataset(dataset_list, transform=train_transform, target_transform=target_transform, is_train=is_train)
    data_loaders = []
    random_seed = 10

    for dataset in datasets:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split=int(math.ceil(dataset_size * 1.0 / (size-1)))
        if is_train:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
            offset = split * (rank-1)
            worker_indices = indices[offset: offset + split]
        # worker_sampler=SubsetRandomSampler(worker_indices)
        # master_sampler = torch.utils.data.RandomSampler(dataset)
        sampler=SubsetRandomSampler(worker_indices) if is_train else torch.utils.data.SequentialSampler(dataset)

        batch_size = cfg.SOLVER.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)

        data_loader = DataLoader(dataset, num_workers=cfg.DATA_LOADER.NUM_WORKERS, batch_sampler=batch_sampler,
                                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY, collate_fn=BatchCollator(is_train))
        data_loaders.append(data_loader)
    return data_loaders[0]


def run_worker_process(cfg,model,comm,rank,size,args):
    logger = logging.getLogger('SSD.trainer for rank {}'.format(rank))
    logger.info("Start training ...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:{}".format(rank))
    print('device of rank{} is {}'.format(rank,device))
    model.to(device)
    model.train()
    train_loader = my_data_loader(cfg,  rank, size, is_train=True)
    # lr = cfg.SOLVER.LR * args.num_gpus  # scale by num gpus
    lr = cfg.SOLVER.LR
    optimizer = make_optimizer(cfg, model, lr)
    milestones = [step // args.num_gpus for step in cfg.SOLVER.LR_STEPS]
    scheduler = make_lr_scheduler(cfg, optimizer, milestones)
    for iteration, (images, targets, img_ids) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        # print('img_id for rank {}'.format(rank),img_ids)
        # print('images',images)
        # print('targets',targets)
        loss_dict = model(images, targets=targets)
        loss = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    comm.send(model.cpu().state_dict(), 0)
    print('rank {} finished sent state'.format(rank))
    # torch.cuda.empty_cache()



def run_master_process(model,comm,rank,size):
    print("* Waiting for {0} training processes to finish...".format(size - 1))
    state_dicts = []
    # device = torch.device("cuda:0")
    for p in range(size - 1):
        state_dicts.append(comm.recv())
        print("(Received a trained model from process {0} of {1} workers...)".format(p + 1, size - 1))
    print("* Averaging models...")
    avg_state_dict = OrderedDict()
    for key in state_dicts[0].keys():
        avg_state_dict[key] = sum([sd[key] for sd in state_dicts]) / float(size - 1)
        # print([sd[key] for sd in state_dicts])
    device = torch.device("cuda:0")
    model.load_state_dict(avg_state_dict)
    model.to(device)
    # --- run validation loop:
    model.eval()
    with torch.no_grad():
        eval_results = []
        val_loader = my_data_loader(cfg, rank, size, is_train=False)
        predictions = compute_on_dataset(model, val_loader, device)
        image_ids = list(sorted(predictions.keys()))
        predictions = [predictions[i] for i in image_ids]
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        dataset = val_loader.dataset
        eval_result = evaluate(dataset=dataset, predictions=predictions, output_dir=output_folder)
        eval_results.append(eval_result)

        # print("* Mean eval results of averaged model: {}".format(np.mean(eval_results)))


def dist_train(cfg,args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # set the random seed to be different for each process:
    torch.manual_seed(rank)
    # decide what to do based on rank:
    if rank == 0:
        # build a fresh model:
        model = build_detection_model(cfg)
        # device = torch.device(cfg.MODEL.DEVICE)
        device = torch.device("cuda:0")
        # model = nn.DataParallel(model)
        model.to(device)
        # loop over some number of epochs:
        for t in range(args.nepochs):
            print("[ = = = = = Epoch {} = = = = = ]".format(t))
            [comm.send(model.state_dict(), k) for k in range(1, size)]
            run_master_process(model, comm, rank, size)
    else:
        for t in range(args.nepochs):
            model = build_detection_model(cfg)
            # device = torch.device(cfg.MODEL.DEVICE)
            device = torch.device("cuda:{}".format(rank))
            # model = nn.DataParallel(model)
            model.to(device)
            model.load_state_dict(comm.recv())
            run_worker_process(cfg,model,comm,rank,size,args)




def main():
    parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With PyTorch')
    parser.add_argument(
        "--config_file",
        default="configs/vgg_ssd300_voc0712.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--nepochs", dest="nepochs", default=300,
                        help="Number of epochs (times to loop through the dataset).")
    args = parser.parse_args()
    num_gpus = 4
    args.num_gpus = num_gpus

    if torch.cuda.is_available():
        # This flag allows you to enable the inbuilt cudnn auto-tuner to
        # find the best algorithm to use for your hardware.
        torch.backends.cudnn.benchmark = True


    cfg.merge_from_file(args.config_file)

    cfg.freeze()

    if cfg.OUTPUT_DIR:
        mkdir(cfg.OUTPUT_DIR)

    logger = setup_logger("SSD", dist_util.get_rank(), cfg.OUTPUT_DIR)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))
    dist_train(cfg,args)




if __name__ == '__main__':
    main()
