#!/usr/bin/env python

import os
import subprocess
from os.path import *
from types import SimpleNamespace

import colorama
import numpy as np
import setproctitle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from . import datasets, losses, models
    from .utils import flow_utils, tools
except:
    import datasets, losses, models
    from utils import flow_utils, tools

# fp32 copy of parameters for update
global param_copy


def infer_flownet(in_path, out_path, reverse, downscale_factor=1):
    args = SimpleNamespace(
        model="FlowNet2",
        reverse=reverse,
        downscale_factor=downscale_factor,
        start_epoch=1,
        total_epochs=10000,
        batch_size=8,
        train_n_batches=-1,
        crop_size=[256, 256],
        gradient_clip=None,
        schedule_lr_frequency=0,
        schedule_lr_fraction=10,
        rgb_max=255.,
        number_workers=8,
        number_gpus=-1,
        no_cuda=False,
        seed=1,
        name='run',
        in_path=in_path,
        save=out_path,
        validation_frequency=5,
        validation_n_batches=-1,
        render_validation=False,
        inference=True,
        inference_visualize=True,
        inference_size=[-1, -1],
        inference_batch_size=1,
        inference_n_batches=-1,
        save_flow=True,
        resume='./FlowNet2_checkpoint.pth.tar',
        log_frequency=1,
        skip_training=False,
        skip_validation=False,
        fp16=False,
        fp16_scale=1024.,
        loss='L1Loss',
        optimizer='Adam',
        training_dataset='MpiSintelFinal',
        root='./MPI-Sintel/flow/training',
        validation_dataset='MpiSintelClean',
        inference_dataset='MpiSintelClean',
        IGNORE=False)
    print(args.in_path)
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        if args.number_gpus < 0: args.number_gpus = torch.cuda.device_count()

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

        if args.inference:
            args.skip_validation = True
            args.skip_training = True
            args.total_epochs = 1
            args.inference_dir = "{}/inference".format(args.save)

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load the dataset class with parameters passed in via "--argument_[param]=[value]" arguments
    with tools.TimerBlock("Initializing Datasets") as block:
        args.effective_batch_size = args.batch_size * args.number_gpus
        args.effective_inference_batch_size = args.inference_batch_size * args.number_gpus
        args.effective_number_workers = args.number_workers * args.number_gpus
        gpuargs = {'num_workers': args.effective_number_workers,
                   'pin_memory': True,
                   'drop_last': True} if args.cuda else {}
        inf_gpuargs = gpuargs.copy()
        inf_gpuargs['num_workers'] = args.number_workers

        inference_dataset = datasets.ImagesFromFolder(args, is_cropped=False, is_reversed=args.reverse,
                                                      root=args.in_path)
        block.log(
            'Inference Input: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][0]])))
        block.log(
            'Inference Targets: {}'.format(' '.join([str([d for d in x.size()]) for x in inference_dataset[0][1]])))
        inference_loader = DataLoader(inference_dataset, batch_size=args.effective_inference_batch_size,
                                      shuffle=False, **inf_gpuargs)

    # Dynamically load model and loss class with parameters passed in via "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class ModelAndLoss(nn.Module):
            def __init__(self, args):
                super(ModelAndLoss, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data, target, inference=False):
                output = self.model(data)

                loss_values = self.loss(output, target)

                if not inference:
                    return loss_values
                else:
                    return loss_values, output

        model_and_loss = ModelAndLoss(args)

        # assing to cuda or wrap with dataparallel, model and loss
        if args.cuda and (args.number_gpus > 0) and args.fp16:
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            model_and_loss = model_and_loss.cuda().half()
            torch.cuda.manual_seed(args.seed)
        elif args.cuda and args.number_gpus > 0:
            model_and_loss = model_and_loss.cuda()
            model_and_loss = nn.parallel.DataParallel(model_and_loss, device_ids=list(range(args.number_gpus)))
            torch.cuda.manual_seed(args.seed)
        else:
            block.log('CUDA not being used')
            torch.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        if args.resume and os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            if not args.inference:
                args.start_epoch = checkpoint['epoch']
            model_and_loss.module.model.load_state_dict(checkpoint['state_dict'])
        elif args.resume and args.inference:
            block.log("No checkpoint found at '{}'".format(args.resume))
            quit()
        else:
            block.log("Random initialization")

        if not os.path.exists(args.save):
            os.makedirs(args.save)

    # Reusable function for inference
    def inference(args, data_loader, model, offset=0):
        model.eval()
        if args.save_flow or args.render_validation:
            flow_folder = out_path  # "./output/flo_rev" if args.reverse else "./output/flo"
            if not os.path.exists(flow_folder):
                os.makedirs(flow_folder)

        # visualization folder
        if args.inference_visualize:
            flow_vis_folder = out_path + "/" + "png/"
            if not os.path.exists(flow_vis_folder):
                os.makedirs(flow_vis_folder)

        args.inference_n_batches = np.inf if args.inference_n_batches < 0 else args.inference_n_batches

        progress = tqdm(data_loader, ncols=100, total=np.minimum(len(data_loader), args.inference_n_batches),
                        desc='Inferencing ',
                        leave=True, position=offset)

        statistics = []
        total_loss = 0
        ph, pw = inference_dataset.ph, inference_dataset.pw
        for batch_idx, (data, target) in enumerate(progress):
            if args.cuda:
                data, target = [d.cuda(non_blocking=True) for d in data], [t.cuda(non_blocking=True) for t in target]
            data, target = [Variable(d) for d in data], [Variable(t) for t in target]

            # when ground-truth flows are not available for inference_dataset,
            # the targets are set to all zeros. thus, losses are actually L1 or L2 norms of compute optical flows,
            # depending on the type of loss norm passed in
            with torch.no_grad():
                losses, output = model(data[0], target[0], inference=True)

            losses = [torch.mean(loss_value) for loss_value in losses]
            loss_val = losses[0]  # Collect first loss for weight update
            total_loss += loss_val.item()
            loss_values = [v.item() for v in losses]

            statistics.append(loss_values)
            # import IPython; IPython.embed()
            if args.save_flow or args.render_validation:
                for i in range(args.inference_batch_size):
                    _pflow = output[i].data.cpu().numpy().transpose(1, 2, 0)
                    if ph != 0:
                        _pflow = _pflow[ph:-ph, :, :]
                    if pw != 0:
                        _pflow = _pflow[:, pw:-pw, :]
                    flow_utils.writeFlow(join(flow_folder, '%06d.flo' % (batch_idx * args.inference_batch_size + i)),
                                         _pflow)

                    # You can comment out the plt block in visulize_flow_file() for real-time visualization
                    if args.inference_visualize:
                        flow_utils.visulize_flow_file(
                            join(flow_folder, '%06d.flo' % (batch_idx * args.inference_batch_size + i)),
                            flow_vis_folder)

            progress.update(1)
            if batch_idx == (args.inference_n_batches - 1):
                break
        progress.close()
        return

    progress = tqdm(list(range(args.start_epoch, args.total_epochs + 1)), miniters=1, ncols=100,
                    desc='Overall Progress', leave=True, position=0)
    offset = 1

    for _ in progress:
        inference(args=args, data_loader=inference_loader, model=model_and_loss, offset=offset)
        offset += 1
    print("\n")


if __name__ == '__main__':
    infer_flownet("C:\\Users\\ZG\\Desktop\\masks", "./out", reverse=False)
