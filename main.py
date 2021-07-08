import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils import get_loops, get_dataset, get_network, get_eval_pool, match_loss, get_time, TensorDataset, epoch, \
    vizualize, eval_synthetic


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: same as training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300,
                        help='epochs to train a model with synthetic data before testing on real data')
    parser.add_argument('--K', type=int, default=1000, help='number of model random initializations to try')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--ink', type=str, default='noise',
                        help='initialization of synthetic data, noise/real: initialize from random noise or real images. The two initializations will get similar performances.')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard/', help='distance metric')
    # For speeding up, we can decrease the K and epoch_eval_train, which will not cause significant performance decrease.
    
    args = parser.parse_args()
    
    # loop_net = ςθ, loop_syn = ςS
    args.T, args.loop_net = get_loops(args.ipc)
    args.loop_syn = 1
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    writer = SummaryWriter(log_dir=args.tensorboard_dir)
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    
    eval_it_pool = np.arange(0, args.K + 1, (args.K + 1) // 2).tolist() if args.eval_mode == 'S' else [
        args.K]  # evaluate on synthetic data at these iterations
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = get_dataset(args.dataset,
                                                                                             args.data_path)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=2)
    model_eval_pool = get_eval_pool(args.eval_mode,
                                    args.model)  # list of models to evaluate on synthetic data at iterations specified in eval_it_pool
    
    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []
    
    data_save = []
    
    for exp in range(args.num_exp):
        print('\n================== Exp %d ==================\n ' % exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)
        
        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]
        
        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(args.device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)
        
        for c in range(num_classes):
            print('class c = %d: %d real images' % (c, len(indices_class[c])))
        
        def get_real_images(c, batch_size):  # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:batch_size]
            return images_all[idx_shuffle], labels_all[idx_shuffle]
        
        for ch in range(channel):
            print('real images channel %d, mean = %.4f, std = %.4f' % (
            ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))
        
        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.long,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]
        
        if args.ink == 'real':
            print('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc], label_syn.data[
                                                                 c * args.ipc:(c + 1) * args.ipc] = get_real_images(c,
                                                                                                                    args.ipc).detach().data
        else:
            print('initialize synthetic data from random noise')
        
        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        print('%s training begins' % get_time())
        
        for k in range(args.K + 1):
            
            ''' Train on synthetic data and test on real data '''
            if k in eval_it_pool:
                # train a random model on synthetic data for fixed number of epochs (no validation) and test on test data
                accs_all_exps = eval_synthetic(k, model_eval_pool, accs_all_exps, image_syn, label_syn, testloader,
                                               args, channel, num_classes, im_size)
                
                ''' visualize and save '''
                vizualize(image_syn, channel, std, mean, args, exp, k)
            
            ''' Train synthetic data '''
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net, momentum=0.5)  # network optimizer
            optimizer_net.zero_grad()
            loss_avg = 0
            
            for t in range(args.T):
                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with real data batch ahead.
                # This would make the training with BatchNorm layers easier.
                
                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name():  # BatchNorm
                        BN_flag = True
                if BN_flag:
                    img_real, _ = torch.cat([get_real_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train()  # for updating the mu, sigma of BatchNorm
                    _ = net(img_real)  # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  # BatchNorm
                            module.eval()  # fix mu and sigma of every BatchNorm layer
                
                ''' update synthetic data '''
                loss = torch.tensor(0.0, device=args.device)
                for c in range(num_classes):
                    img_real, lab_real = get_real_images(c, args.batch_real)
                    # lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    output_real = net(img_real)
                    loss_real = criterion(output_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, net_parameters)
                    
                    # note we only detach real image gradients but not synthesized ones
                    gw_real = list((grad.detach().clone() for grad in gw_real))
                    
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    
                    # create_graph is set to True since this computational graph is used again when calling loss.backward()
                    # to compute gradients with respect to all leaf variables including img_syn
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)
                    
                    loss += match_loss(gw_syn, gw_real, args)
                
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()
                
                if t == args.T - 1:
                    break
                
                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                          num_workers=0)
                for il in range(args.loop_net):
                    epoch('train', trainloader, net, optimizer_net, criterion, None, args.device)
            
            loss_avg /= (num_classes * args.T)
            
            if k % 10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), k, loss_avg))
                writer.add_scalar(f'exp_{exp}/loss', loss_avg, k)
            
            if k == args.K:  # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, },
                           os.path.join(args.save_path, 'res_%s_%s_%dipc.pt' % (args.dataset, args.model, args.ipc)))
    
    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (
        args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))


if __name__ == '__main__':
    main()
