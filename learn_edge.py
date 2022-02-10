"""Unified interface to all dynamic graph model experiments"""
import math
import logging
import time
import random
import sys
import argparse

import torch
from torch.profiler import profile, ProfilerActivity
import pandas as pd
import numpy as np

from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

from module import TGAN
from graph import NeighborFinder
from utils import EarlyStopMonitor, RandEdgeSampler


def eval_one_epoch(model: TGAN, sampler, src_list, dst_list, ts_list, num_neighbors):
    _val_acc, _val_ap, _val_f1, _val_auc = [], [], [], []
    with torch.no_grad():
        model = model.eval()
        TEST_BATCH_SIZE = 100
        num_test_instance = len(src_list)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)
        for batch_idx in range(num_test_batch):
            st_idx = batch_idx * TEST_BATCH_SIZE
            ed_idx = min(num_test_instance - 1, st_idx + TEST_BATCH_SIZE)
            
            if st_idx == ed_idx:
                break
            
            _src_l_cut = src_list[st_idx:ed_idx]
            _dst_l_cut = dst_list[st_idx:ed_idx]
            _ts_l_cut = ts_list[st_idx:ed_idx]
            
            elem_size = len(_src_l_cut)
            _src_l_fake, _dst_l_fake = sampler.sample(elem_size)
            
            try:
                pos_p, neg_p = model.contrast(_src_l_cut, _dst_l_cut, _dst_l_fake, _ts_l_cut, num_neighbors)
            except IndexError as e:
                print(_src_l_cut.shape, _dst_l_cut.shape, _dst_l_fake.shape, _ts_l_cut.shape)
                print(st_idx, ed_idx, _src_l_cut, _dst_l_cut, _ts_l_cut)
                raise e
            
            _pred_score = np.concatenate([pos_p.cpu().numpy(), neg_p.cpu().numpy()])
            _pred_label = _pred_score > 0.5
            _true_label = np.concatenate([np.ones(elem_size), np.zeros(elem_size)])
            
            _val_acc.append((_pred_label == _true_label).mean())
            _val_ap.append(average_precision_score(_true_label, _pred_score))
            # val_f1.append(f1_score(true_label, pred_label))
            _val_auc.append(roc_auc_score(_true_label, _pred_score))
    return np.mean(_val_acc), np.mean(_val_ap), np.mean(_val_f1), np.mean(_val_auc)


def run(args):
    BATCH_SIZE = args.bs
    NUM_NEIGHBORS = args.n_degree
    NUM_NEG = 1
    NUM_EPOCH = args.n_epoch
    NUM_HEADS = args.n_head
    DROP_OUT = args.drop_out
    GPU = args.gpu
    UNIFORM = args.uniform
    USE_TIME = args.time
    AGG_METHOD = args.agg_method
    ATTN_MODE = args.attn_mode
    SEQ_LEN = NUM_NEIGHBORS
    DATA = args.data
    NUM_LAYER = args.n_layer
    LEARNING_RATE = args.lr
    NODE_DIM = args.node_dim
    TIME_DIM = args.time_dim
    
    MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}.pth'
    
    def get_checkpoint_path(_epoch):
        return f'./saved_checkpoints/{args.prefix}-{args.agg_method}-{args.attn_mode}-{DATA}-{_epoch}.pth'
    
    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    
    # Load data and train val test split
    g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
    e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
    n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
    
    val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))  # train: 70%, val: 15%, test: 15%
    
    src_l = g_df.u.values  # source list
    dst_l = g_df.i.values  # destination list
    e_idx_l = g_df.idx.values  # edge index list
    label_l = g_df.label.values  # node label list
    ts_l = g_df.ts.values  # timestamp list
    
    max_idx = max(src_l.max(), dst_l.max())
    
    random.seed(2020)
    
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    num_total_unique_nodes = len(total_node_set)
    
    num_mask = int(0.1 * num_total_unique_nodes)
    mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), num_mask))
    mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
    mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
    none_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
    
    valid_train_flag = (ts_l <= val_time) * (none_node_flag > 0)
    
    train_src_l = src_l[valid_train_flag]
    train_dst_l = dst_l[valid_train_flag]
    train_ts_l = ts_l[valid_train_flag]
    train_e_idx_l = e_idx_l[valid_train_flag]
    train_label_l = label_l[valid_train_flag]
    
    # define the new nodes sets for testing inductiveness of the model
    train_node_set = set(train_src_l).union(train_dst_l)
    assert(len(train_node_set - mask_node_set) == len(train_node_set))
    new_node_set = total_node_set - train_node_set
    
    # select validation and test dataset
    valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
    valid_test_flag = ts_l > test_time
    
    is_new_node_edge = np.array([(a in new_node_set or b in new_node_set) for a, b in zip(src_l, dst_l)])
    nn_val_flag = valid_val_flag * is_new_node_edge
    nn_test_flag = valid_test_flag * is_new_node_edge

    # validation and test with all edges
    val_src_l = src_l[valid_val_flag]
    val_dst_l = dst_l[valid_val_flag]
    val_ts_l = ts_l[valid_val_flag]
    
    test_src_l = src_l[valid_test_flag]
    test_dst_l = dst_l[valid_test_flag]
    test_ts_l = ts_l[valid_test_flag]
    # test_e_idx_l = e_idx_l[valid_test_flag]
    # test_label_l = label_l[valid_test_flag]
    
    # validation and test with edges that at least has one new node (not in training set)
    nn_val_src_l = src_l[nn_val_flag]
    nn_val_dst_l = dst_l[nn_val_flag]
    nn_val_ts_l = ts_l[nn_val_flag]
    # nn_val_e_idx_l = e_idx_l[nn_val_flag]
    # nn_val_label_l = label_l[nn_val_flag]
    
    nn_test_src_l = src_l[nn_test_flag]
    nn_test_dst_l = dst_l[nn_test_flag]
    nn_test_ts_l = ts_l[nn_test_flag]
    # nn_test_e_idx_l = e_idx_l[nn_test_flag]
    # nn_test_label_l = label_l[nn_test_flag]
    
    # Initialize the data structure for graph and edge sampling
    # build the graph for fast query
    # graph only contains the training data (with 10% nodes removal)
    adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        adj_list[src].append((dst, eidx, ts))
        adj_list[dst].append((src, eidx, ts))
    train_ngh_finder = NeighborFinder(adj_list, uniform=UNIFORM)
    
    # full graph with all the data for the test and validation purpose
    full_adj_list = [[] for _ in range(max_idx + 1)]
    for src, dst, eidx, ts in zip(src_l, dst_l, e_idx_l, ts_l):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, uniform=UNIFORM)
    
    train_rand_sampler = RandEdgeSampler(train_src_l, train_dst_l)
    val_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_val_rand_sampler = RandEdgeSampler(nn_val_src_l, nn_val_dst_l)
    test_rand_sampler = RandEdgeSampler(src_l, dst_l)
    nn_test_rand_sampler = RandEdgeSampler(nn_test_src_l, nn_test_dst_l)
    
    # Model initialize
    device = torch.device('cuda:{}'.format(GPU))
    tgan = TGAN(train_ngh_finder, n_feat, e_feat,
                num_layers=NUM_LAYER, use_time=USE_TIME, agg_method=AGG_METHOD, attn_mode=ATTN_MODE,
                seq_len=SEQ_LEN, n_head=NUM_HEADS, drop_out=DROP_OUT, node_dim=NODE_DIM, time_dim=TIME_DIM)
    optimizer = torch.optim.Adam(tgan.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()
    tgan = tgan.to(device)
    
    num_instance = len(train_src_l)
    num_batch = math.ceil(num_instance / BATCH_SIZE)
    
    logger.info('num of training instances: {}'.format(num_instance))
    logger.info('num of batches per epoch: {}'.format(num_batch))
    idx_list = np.arange(num_instance)
    np.random.shuffle(idx_list)
    early_stopper = EarlyStopMonitor()
    
    start_time = time.time()
    
    for epoch in range(NUM_EPOCH):
        # training use only training graph
        tgan.ngh_finder = train_ngh_finder
        acc, ap, f1, auc, m_loss = [], [], [], [], []
        np.random.shuffle(idx_list)
        logger.info('start {} epoch'.format(epoch))
        st = time.time()
        for k in range(num_batch):
            
            if (k + 1) % 10 == 0:
                tm = time.time()
                logger.info("Batch {} / {}, {:.2f} [s]".format(k + 1, num_batch, tm - st))
            
            s_idx = k * BATCH_SIZE
            e_idx = min(num_instance - 1, s_idx + BATCH_SIZE)
            
            if s_idx == e_idx:
                break
            
            src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            ts_l_cut = train_ts_l[s_idx:e_idx]
            label_l_cut = train_label_l[s_idx:e_idx]
            size = len(src_l_cut)
            src_l_fake, dst_l_fake = train_rand_sampler.sample(size)
            
            with torch.no_grad():
                pos_label = torch.ones(size, dtype=torch.float, device=device)
                neg_label = torch.zeros(size, dtype=torch.float, device=device)
            
            optimizer.zero_grad()
            tgan = tgan.train()
            pos_prob, neg_prob = tgan.contrast(src_l_cut, dst_l_cut, dst_l_fake, ts_l_cut, NUM_NEIGHBORS)
            
            loss = criterion(pos_prob, pos_label)
            loss += criterion(neg_prob, neg_label)
            
            loss.backward()
            optimizer.step()
            # get training results
            with torch.no_grad():
                tgan = tgan.eval()
                pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
                pred_label = pred_score > 0.5
                true_label = np.concatenate([np.ones(size), np.zeros(size)])
                acc.append((pred_label == true_label).mean())
                ap.append(average_precision_score(true_label, pred_score))
                m_loss.append(loss.item())
                auc.append(roc_auc_score(true_label, pred_score))
        
        tm = time.time()
        print("Processed {} batches in {:.2f} [s]".format(num_batch, tm - st))
        
        # validation phase use all information
        tgan.ngh_finder = full_ngh_finder
        # Validation for old (known) nodes
        val_acc, val_ap, val_f1, val_auc = eval_one_epoch(tgan, val_rand_sampler, val_src_l,
                                                          val_dst_l, val_ts_l, NUM_NEIGHBORS)
        # Validation for new (unknown) nodes
        nn_val_acc, nn_val_ap, nn_val_f1, nn_val_auc = eval_one_epoch(tgan, val_rand_sampler, nn_val_src_l,
                                                                      nn_val_dst_l, nn_val_ts_l, NUM_NEIGHBORS)
        
        ed = time.time()
        print("Eval epoch {:.2f} [s]".format(ed - tm))
        logger.info('epoch: {}, {:.2f} [s]:'.format(epoch, ed - st))
        logger.info('Epoch mean loss: {:.4f}'.format(np.mean(m_loss)))
        logger.info('train acc: {:.4f}, val acc: {:.4f}, new node val acc: {:.4f}'.format(np.mean(acc), val_acc, nn_val_acc))
        logger.info('train auc: {:.4f}, val auc: {:.4f}, new node val auc: {:.4f}'.format(np.mean(auc), val_auc, nn_val_auc))
        logger.info('train ap: {:.4f}, val ap: {:.4f}, new node val ap: {:.4f}'.format(np.mean(ap), val_ap, nn_val_ap))
        # logger.info('train f1: {}, val f1: {}, new node val f1: {}'.format(np.mean(f1), val_f1, nn_val_f1))
        
        if early_stopper.early_stop_check(val_ap):
            logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_model_path = get_checkpoint_path(early_stopper.best_epoch)
            tgan.load_state_dict(torch.load(best_model_path))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            tgan.eval()
            break
        else:
            torch.save(tgan.state_dict(), get_checkpoint_path(epoch))
    
    end_time = time.time()
    print("{} epochs in {:.2f}".format(epoch, end_time - start_time))
    
    # testing phase use all information
    tgan.ngh_finder = full_ngh_finder
    # Testing for old (known) nodes
    test_acc, test_ap, test_f1, test_auc = eval_one_epoch(tgan, test_rand_sampler, test_src_l,
                                                          test_dst_l, test_ts_l, NUM_NEIGHBORS)
    # Testing for new (unknown) nodes
    nn_test_acc, nn_test_ap, nn_test_f1, nn_test_auc = eval_one_epoch(tgan, nn_test_rand_sampler, nn_test_src_l,
                                                                      nn_test_dst_l, nn_test_ts_l, NUM_NEIGHBORS)
    
    logger.info('Test statistics: Old nodes -- acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(test_acc, test_auc, test_ap))
    logger.info('Test statistics: New nodes -- acc: {:.4f}, auc: {:.4f}, ap: {:.4f}'.format(nn_test_acc, nn_test_auc, nn_test_ap))
    
    logger.info('Saving TGAN model')
    torch.save(tgan.state_dict(), MODEL_SAVE_PATH)
    logger.info('TGAN models saved')


if __name__ == "__main__":
    # Argument and global variables
    parser = argparse.ArgumentParser('Interface for TGAT experiments on link predictions')
    parser.add_argument('-d', '--data', type=str, help='data sources to use', default='wikipedia',
                        choices=["reddit", "wikipedia", "mooc", "lastfm"])
    parser.add_argument('--bs', type=int, default=200, help='batch_size')
    parser.add_argument('--prefix', type=str, default='', help='prefix to name the checkpoints')
    parser.add_argument('--n_degree', type=int, default=20, help='number of neighbors to sample')
    parser.add_argument('--n_head', type=int, default=2, help='number of heads used in attention layer')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--n_layer', type=int, default=2, help='number of network layers')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout probability')
    parser.add_argument('--gpu', type=int, default=0, help='idx for the gpu to use')
    parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
    parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
    parser.add_argument('--agg_method', type=str, choices=['attn', 'lstm', 'mean'],
                        help='local aggregation method', default='attn')
    parser.add_argument('--attn_mode', type=str, choices=['prod', 'map'], default='prod',
                        help='use dot product attention or mapping based')
    parser.add_argument('--time', type=str, choices=['time', 'pos', 'empty'],
                        help='how to use time information', default='time')
    parser.add_argument('--uniform', action='store_true', help='take uniform sampling from temporal neighbors')
    parser.add_argument("--prof", help="PyTorch profiler name", type=str)
    
    try:
        _args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    prof_name = _args.prof
    if prof_name is None:
        run(_args)
    else:
        prof_json = "{}.json".format(prof_name)
        prof_txt = "{}.log".format(prof_name)
        with profile(use_cuda=True, activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            run(_args)
        with open(prof_txt, "w") as wf:
            wf.write(str(prof.key_averages().table(sort_by="self_cpu_time_total")))
        # prof.export_chrome_trace(prof_json)
