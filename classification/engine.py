# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from logging import critical
import math
from typing import Iterable, Optional
import torch
import torch.nn.functional as F
from timm.data import Mixup
from timm.utils import accuracy, ModelEma
from timm.loss import SoftTargetCrossEntropy


import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    wandb_logger=None, start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, use_amp=False):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(samples)
                loss = criterion(output, targets)
        else:  # full precision
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):  # this could trigger if using AMP
            print("Loss is {}, stopping training".format(loss_value))
            assert math.isfinite(loss_value)

        if use_amp:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
        else:  # full precision
            loss /= update_freq
            loss.backward()

            if max_norm is not None:  # clip grad
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_norm = utils.get_grad_norm_(model.parameters())

            if (data_iter_step + 1) % update_freq == 0:
                optimizer.step()
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if mixup_fn is None:
            class_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            class_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value)
        if use_amp:
            metric_logger.update(grad_norm=grad_norm)

        if log_writer is not None:
            log_writer.update(loss=loss_value, head="loss")
            log_writer.update(class_acc=class_acc, head="loss")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            # if use_amp:
            #    log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")
            log_writer.set_step()

        if wandb_logger:
            wandb_logger._wandb.log({
                'Rank-0 Batch Wise/train_loss': loss_value,
                'Rank-0 Batch Wise/train_max_lr': max_lr,
                'Rank-0 Batch Wise/train_min_lr': min_lr
            }, commit=False)
            if class_acc:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_class_acc': class_acc}, commit=False)
            if use_amp:
                wandb_logger._wandb.log({'Rank-0 Batch Wise/train_grad_norm': grad_norm}, commit=False)
            wandb_logger._wandb.log({'Rank-0 Batch Wise/global_train_step': it})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, use_amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
                loss = criterion(output, target)
        else:
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class VarianceCollateFN:
    def __init__(self, standard_transform, variance_transforms: dict):
        self.standard_transform = standard_transform
        self.variance_transforms = variance_transforms

    def __call__(self, data_list):
        standard_img = []
        variance_img = {}
        gold = []

        for img, target in data_list:
            standard_img.append(self.standard_transform(img))
            gold.append(target)

            for n, t in self.variance_transforms.items():
                if n not in variance_img:
                    variance_img[n] = []

                variance_img[n].append(t(img))

        for key in variance_img:
            variance_img[key] = torch.stack(variance_img[key], dim=0)

        return {
            'standard_img': torch.stack(standard_img, dim=0),
            'variance_img': variance_img,
            'gold': torch.LongTensor(gold),
        }


@torch.no_grad()
def evaluate_invariance(data_loader, model, device, use_amp=False):
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = SoftTargetCrossEntropy()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Invariance Test:'

    # switch to evaluation mode
    model.eval()
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch['standard_img']
        images = images.to(device, non_blocking=True)
        gold_target = batch['gold'].to(device, non_blocking=True)

        if use_amp:
            with torch.cuda.amp.autocast():
                pred_logits = model(images)
        else:
            pred_logits = model(images)

        pred_target = F.softmax(pred_logits, dim=-1)
        pred_label = pred_logits.max(dim=1)[1]

        batch_size = images.shape[0]
        for variance_name, transformed_images in batch['variance_img'].items():
            transformed_images = transformed_images.to(device)
            if use_amp:
                with torch.cuda.amp.autocast():
                    output = model(transformed_images)
                    loss = criterion(output, pred_target)
            else:
                output = model(transformed_images)
                loss = criterion(output, pred_target)

            metric_logger.meters[f'{variance_name} loss'].update(loss.item(), n=batch_size)

            acc1, acc5 = accuracy(output, gold_target, topk=(1, 5))
            consistency = accuracy(output, pred_label)[0]
            metric_logger.meters[f'{variance_name} acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters[f'{variance_name} acc5'].update(acc5.item(), n=batch_size)
            metric_logger.meters[f'{variance_name} consistency'].update(consistency.item(), n=batch_size)

        acc1, acc5 = accuracy(pred_logits, gold_target, topk=(1, 5))
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Eval Results')
    for key, value in metric_logger.meters.items():
        print(f'\t{key}: {value}')

    return metric_logger.meters
