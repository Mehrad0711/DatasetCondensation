import os
import copy
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForSequenceClassification, AdamW, get_constant_schedule_with_warmup, \
    get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from torchvision.utils import save_image

import logging


def initialize_logger():
    # set up file logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    return logger

logger = logging.getLogger(__name__)


def freeze_params(model):
    for name, par in model.named_parameters():
        if 'classifier' in name:
            continue
        par.requires_grad = False


def unfreeze_params(model):
    for par in model.parameters():
        par.requires_grad = True


def get_transformer_learning_rate(i, *, dimension, warmup):
    i += 1
    return 1.0 / math.sqrt(dimension) * min(1 / math.sqrt(i), i / (warmup * math.sqrt(warmup)))


def get_sgd_learning_rate(i, *, warmup):
    i += 1
    return min(math.sqrt(warmup) / math.sqrt(i), i / warmup)


def init_opt(args, lr_multiply, params):
    if args.optimizer == 'adam':
        # Adam with transformer schedule has a different set of default hyperparameters:
        if args.lr_schedule == 'transformer':
            opt = torch.optim.Adam(
                params, lr=lr_multiply, betas=(0.9, 0.98), eps=1e-9, weight_decay=args.weight_decay
            )
        else:
            opt = torch.optim.Adam(
                params, lr=lr_multiply, betas=(args.beta0, 0.999), weight_decay=args.weight_decay
            )
    elif args.optimizer == 'adamw':
        opt = AdamW(params, lr=lr_multiply, weight_decay=args.weight_decay)
    elif args.optimizer == 'radam':
        import radam
        opt = radam.RAdam(params, lr=lr_multiply, betas=(args.beta0, 0.999),
                          weight_decay=args.weight_decay)
    else:
        assert args.optimizer == 'sgd'
        opt = torch.optim.SGD(params, lr=lr_multiply, weight_decay=args.weight_decay)

    if args.lr_schedule == 'transformer':
        lr_lambda = partial(get_transformer_learning_rate, dimension=args.dimension, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    elif args.lr_schedule == 'constant':
        scheduler = get_constant_schedule_with_warmup(opt, num_warmup_steps=args.warmup)
    elif args.lr_schedule == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            opt,
            num_training_steps=sum(args.train_iterations) // args.gradient_accumulation_steps,
            num_warmup_steps=args.warmup,
        )
    elif args.lr_schedule == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(
            opt,
            num_training_steps=sum(args.train_iterations) // args.gradient_accumulation_steps,
            num_warmup_steps=args.warmup,
            num_cycles=0.5,
        )
    elif args.lr_schedule == 'sgd':
        lr_lambda = partial(get_sgd_learning_rate, warmup=args.warmup)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    else:
        raise ValueError('Invalid learning rate scheduler.')

    return opt, scheduler


def evaluate_synset(it_eval, net, images_train, labels_train, testloader, learningrate, batchsize_train, param_augment, device, Epoch = 600):
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(learningrate)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batchsize_train, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, param_augment, device)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, param_augment, device)
    logger.info('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test))

    return net, acc_train, acc_test


def eval_synthetic(it, model_eval_pool, accs_all_exps, image_syn, label_syn, testloader, args, num_classes):
    for model_eval in model_eval_pool:
        logger.info('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, K = %d'%(args.model_name, model_eval, it))
        accs = []
        for it_eval in range(args.num_eval):
            net_eval = get_network(args, num_classes)
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args.lr_multiply_net, args.batch_train, args.device, args.epoch_eval_train)
            accs.append(acc_test)
        logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))
        
        if it == args.K: # record the final results
            accs_all_exps[model_eval] += accs
    return accs_all_exps


def vizualize(image_syn, channel, std, mean, args, exp, it):
    save_name = os.path.join(args.save_path, 'vis_%s_%s_%dipc_exp%d_iter%d.png' % (args.dataset, args.model, args.ipc, exp, it))
    image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
    for ch in range(channel):
        image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
    image_syn_vis[image_syn_vis < 0] = 0.0
    image_syn_vis[image_syn_vis > 1] = 1.0
    save_image(image_syn_vis, save_name, nrow=args.ipc)  # Trying normalize = True/False may get better visual effects.
    # The generated images would be slightly different from the visualization results in the paper, because of the initialization and normalization of pixels.


class TensorDataset(Dataset):
    def __init__(self, text, labels): # images: n x c x h x w tensor
        self.text = text.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.text[index], self.labels[index]

    def __len__(self):
        return self.text.shape[0]



def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(args, num_classes):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net = AutoModelForSequenceClassification.from_pretrained('bert-base-cased', num_labels=num_classes)
    # freeze_params(model) # freeze model except for classification layer
    gpu_num = torch.cuda.device_count()
    if gpu_num > 0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(args.device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    return dis_weight



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('DC error: unknown distance function')

    return dis



def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = ipc, 50 // (ipc//10 + 1)
    return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, scheduler, criterion, param_augment, device):
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(device)
    criterion = criterion.to(device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        text = datum[0].float().to(device)
        lab = datum[1].long().to(device)
        n_b = lab.shape[0]

        output = net(inputs_embeds=text)
        logits = output.logits
        loss = criterion(logits, lab)
        acc = np.sum(np.equal(np.argmax(logits.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool
