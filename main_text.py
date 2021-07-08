import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from utils_text import get_loops, get_network, match_loss, get_time, TensorDataset, epoch, initialize_logger, init_opt, \
    get_eval_pool, eval_synthetic, set_seed, get_trainable_params

from torch.utils.tensorboard import SummaryWriter
from transformers import logging

logging.set_verbosity_error()

logger = initialize_logger()


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    
    parser.add_argument('--dataset_name', type=str, default='imdb', help='dataset')
    parser.add_argument('--model_name', type=str, default='bert-base-cased', help='model')
    parser.add_argument('--tpc', type=int, default=1, help='text(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S', help='eval_mode') # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=5, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=300, help='epochs to train a model with synthetic data')
    parser.add_argument('--K', type=int, default=1000, help='number of model random initializations to try')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for synthesized data')
    parser.add_argument('--batch_test', type=int, default=20, help='batch size for testing on real data')
    parser.add_argument('--ink', type=str, default='noise', help='initialization of synthetic data, noise/real: initialize from random noise or real images. The two initializations will get similar performances.')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    # For speeding up, we can decrease the K and epoch_eval_train, which will not cause significant performance decrease.
    
    parser.add_argument('--T', type=int, help='')
    parser.add_argument('--loop_net', type=int, help='')
    parser.add_argument('--warmup', default=40, type=int, help='warmup for learning rate. setting k to 1 disables warmup.')
    parser.add_argument('--grad_clip', default=1.0, type=float, help='gradient clipping')
    parser.add_argument( '--beta0', default=0.9, type=float, help='alternative momentum for Adam (only when not using transformer scheduler), and RAdam', )
    parser.add_argument( '--optimizer', default='adam', choices=['adam', 'adamw', 'sgd', 'radam'], type=str, help='optimizer to use' )
    parser.add_argument( '--lr_schedule', type=str, default='transformer', choices=['transformer', 'constant', 'linear', 'sgd', 'cosine'], help='The learning rate strategy. All of them can be used with or without warmup.', )
    parser.add_argument( '--lr_multiply_syn', default=0.01, type=float, help='Multiplier for the `transformer` learning rate scheduler, constant value for `constant` and maximum value for `linear` and `cosine` schedulers.', )
    parser.add_argument( '--lr_multiply_net', default=0.01, type=float, help='Multiplier for the `transformer` learning rate scheduler, constant value for `constant` and maximum value for `linear` and `cosine` schedulers.', )
    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight L2 regularization')
    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard/', help='distance metric')
    parser.add_argument('--subsample', type=int, default='2000000000')
    parser.add_argument('--seed', type=int, default='1994')


    args = parser.parse_args()
    T, loop_net = get_loops(args.tpc)
    if not args.T:
        args.T = T
    if not args.loop_net:
        args.loop_net = loop_net
    args.loop_syn = 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    writer = SummaryWriter(log_dir=args.tensorboard_dir)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
        
    eval_it_pool = np.arange(0, args.K+1, (args.K+1)//2).tolist() if args.eval_mode == 'S' else [args.K] # The list of Ks when we evaluate models and record results.
    
    dataset = load_dataset(args.dataset_name)
    num_classes = dataset['train'].shape[1]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)

    train, test = list(dataset['train']), list(dataset['test'])
    np.random.shuffle(train)
    np.random.shuffle(test)
    train = train[:args.subsample]
    test = test[:args.subsample]
    train_text, train_labels = [item['text'] for item in train], [item['label'] for item in train]
    test_text, test_labels = [item['text'] for item in test], [item['label'] for item in test]
    
    vocab_size = len(tokenizer.get_vocab())
    
    train_tokenized = tokenizer.batch_encode_plus(train_text, return_tensors='pt', padding='longest', truncation='longest_first')['input_ids']
    train_tokenized = train_tokenized.to(args.device)
    train_labels = torch.tensor(train_labels, dtype=torch.long, device=args.device).view(-1)

    test_tokenized = tokenizer(test_text, return_tensors='pt', padding='longest', truncation='longest_first')['input_ids']
    test_tokenized = test_tokenized.to(args.device)
    test_labels = torch.tensor(test_labels, dtype=torch.long, device=args.device).view(-1)
    
    dst_test = TensorDataset(test_tokenized, test_labels)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=args.batch_test, shuffle=False)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model_name)
    
    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    set_seed(args.seed)

    model = get_network(args, num_classes, 'word_embedding')
    args.dimension = model.config.hidden_size

    ''' organize the real dataset '''
    indices_class = [[] for c in range(num_classes)]

    text_all = train_tokenized
    for i, lab in enumerate(train_labels):
        indices_class[lab].append(i)

    for c in range(num_classes):
        logger.info('class c = %d: %d real images' % (c, len(indices_class[c])))

    def get_real_text(c, batch_size):
        assert indices_class[c]
        # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:batch_size]
        return text_all[idx_shuffle], train_labels[idx_shuffle]

    for exp in range(args.num_exp):
        logger.info(f'\n================== Exp {exp} ==================\n')
        logger.info(f'Hyper-parameters: \n {args.__dict__}')
        logger.info(f'Evaluation model pool: {model_eval_pool}')

        ''' initialize the synthetic data '''
        max_size = 512
        # length is half of maximum data size
        syn_ids = np.random.randint(low=0, high=vocab_size-1, size=(num_classes*args.tpc, max_size//2)).tolist()
        syn_ids = [(item, None) for item in syn_ids]
        syn_tokenized = tokenizer._batch_prepare_for_model(syn_ids)['input_ids']
        syn_tokenized = torch.tensor(syn_tokenized, dtype=torch.long, requires_grad=False, device=args.device)
        word_embedding_layer = model.get_input_embeddings()
        syn_embedded = word_embedding_layer(syn_tokenized).detach().requires_grad_(True)
        label_syn = torch.tensor([np.ones(args.tpc, dtype='int64')*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
        
        # if args.ink == 'real':
        #     logger.info('initialize synthetic data from random real images')
        #     for c in range(num_classes):
        #         syn_tokenized.data[c*args.tpc:(c+1)*args.tpc] = get_real_text(c, args.tpc).detach().data
        # else:
        #     logger.info('initialize synthetic data from random noise')


        ''' training '''
        optimizer_text, scheduler_text = init_opt(args, args.lr_multiply_syn, [syn_embedded, ])
        optimizer_text = torch.optim.SGD([syn_embedded, ], lr=args.lr_multiply_syn, momentum=0.5) # optimizer_text for synthetic data
        optimizer_text.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        logger.info('%s training begins'%get_time())

        for k in range(args.K+1):

            # # ''' Evaluate synthetic data '''
            if k in eval_it_pool:
                accs_all_exps = eval_synthetic(k, model_eval_pool, accs_all_exps, syn_embedded, label_syn, testloader, args, num_classes)

            #     ''' visualize and save '''
            #     vizualize(syn_tokenized, channel, std, mean, args, exp, k)

            ''' Train synthetic data '''
            net = get_network(args, num_classes, 'word_embedding')
            net.train()
            trainable_parameters = get_trainable_params(net)
            optimizer_net, scheduler_net = init_opt(args, args.lr_multiply_net, trainable_parameters)
            optimizer_net.zero_grad()
            loss_avg = 0

            for t in range(args.T):

                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    text_real, lab_real = get_real_text(c, args.batch_real)
                    output_real = net(input_ids=text_real)
                    logits_real = output_real.logits
                    loss_real = criterion(logits_real, lab_real)
                    gw_real = torch.autograd.grad(loss_real, trainable_parameters)
                    gw_real = list((_.detach().clone() for _ in gw_real))

                    text_syn = syn_embedded[c*args.tpc:(c+1)*args.tpc, :, :]
                    lab_syn = torch.ones((args.tpc,), device=args.device, dtype=torch.long) * c
                    output_syn = net(inputs_embeds=text_syn)
                    logits_syn = output_syn.logits
                    loss_syn = criterion(logits_syn, lab_syn)
                    
                    # use allow unused because we pass embeddings for syn instead of ids so word_embedding is not backpropped
                    gw_syn = torch.autograd.grad(loss_syn, trainable_parameters, create_graph=True, allow_unused=False)
                    
                    # pop the first gradient which is None for gw_syn due to ignoring word_embeddings
                    # gw_real = gw_real[1:]
                    # gw_syn = gw_syn[1:]
                    
                    loss += match_loss(gw_syn, gw_real, args)

                optimizer_text.zero_grad()
                loss.backward()
                optimizer_text.step()
                scheduler_text.step()
                loss_avg += loss.item()
                
                if t == args.T - 1:
                    break

                ''' update network '''
                syn_embedded_train, label_syn_train = copy.deepcopy(syn_embedded.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(syn_embedded_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True)
                for il in range(args.loop_net):
                    epoch('train', trainloader, net, optimizer_net, scheduler_net, criterion, args.device)

            loss_avg /= (num_classes*args.T)

            if k % 10 == 0:
                logger.info('%s iter = %04d, loss = %.4f' % (get_time(), k, loss_avg))
                writer.add_scalar(f'exp_{exp}/loss', loss_avg, k)

            # only record the final results
            if k == args.K:
                data_save.append([copy.deepcopy(syn_tokenized.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path, 'res_%s_%s_%dtpc.pt'%(args.dataset_name, args.model_name, args.tpc)))


    logger.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(args.num_exp, args.model_name, len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()


