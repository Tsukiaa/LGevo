# @Time    : 2019-08-10 11:20
# @Author  : Lee_zix
# @Email   : Lee_zix@163.com
# @File    : main.py
# @Software: PyCharm
"""
The entry of the KGEvolve
"""

import argparse
import itertools
import os
import sys
import time
import pickle

import dgl
import numpy as np
import torch
from tqdm import tqdm
import random

sys.path.append("..")
from rgcn import utils
from rgcn.utils import build_sub_graph
from src.rrgcn import RecurrentRGCN
from src.hyperparameter_range import hp_range
import torch.nn.modules.rnn
from collections import defaultdict
from rgcn.knowledge_graph import _read_triplets_as_list


# os.environ['KMP_DUPLICATE_LIB_OK']='True'

def Get_entity_total(train_list):
    entity_list = []
    for i in range(len(train_list)):
        for data in train_list[i]:
            if data[0] not in entity_list:
                entity_list.append(data[0])
            if data[2] not in entity_list:
                entity_list.append(data[2])
    return entity_list

def test(name, model, data, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name,
         static_graph, number, mode):
    """
    :param model: model used to test
    :param history_list:    all input history snap shot list, not include output label train list or valid list
    :param test_list:   test triple snap shot list
    :param num_rels:    number of relations
    :param num_nodes:   number of nodes
    :param use_cuda:
    :param all_ans_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param all_ans_r_list:     dict used to calculate filter mrr (key and value are all int variable not tensor)
    :param model_name:
    :param static_graph
    :param mode
    :return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r
    """
    ranks_raw, ranks_filter, mrr_raw_list, mrr_filter_list = [], [], [], []
    ranks_raw_r, ranks_filter_r, mrr_raw_list_r, mrr_filter_list_r = [], [], [], []

    idx = 0
    if mode == "test":
        # test mode: load parameter form file
        if use_cuda:
            checkpoint = torch.load(model_name, map_location=torch.device(args.gpu))
        else:
            checkpoint = torch.load(model_name, map_location=torch.device('cpu'))
        print("Load Model name: {}. Using best epoch : {}".format(model_name,
                                                                  checkpoint['epoch']))  # use best stat checkpoint
        print("\n" + "-" * 10 + "start testing" + "-" * 10 + "\n")
        model.load_state_dict(checkpoint['state_dict'])
    history_len = len(history_list)
    model.eval()

    t_to_entity = Get_Time_to_Entity(history_list + test_list)

    hr_tail = Get_total_hr_to_tail(history_list + test_list, num_rels)

    for time_idx, test_snap in enumerate(tqdm(test_list)):
        # test_snap = test_list[5]
        output_sample = test_snap

        # 模拟新出现实体
        output_new = test_snap
        output_new = torch.LongTensor(output_new).cuda() if use_cuda else torch.LongTensor(output_new)
        inverse_triples = output_new[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        all_triples = torch.cat([output_new, inverse_triples])
        total_entity = all_triples[:, 0]
        total_relation = all_triples[:, 1]
        h_r_t = Get_head_relation(hr_tail, all_triples, time_idx+history_len, num_nodes)
        zero_tensor_head = np.zeros((len(all_triples), num_nodes))
        for item in h_r_t:
            for data in h_r_t[item]:
                zero_tensor_head[data[0]] = data[1]

        zero_tensor_head[zero_tensor_head == 0] = -100
        zero_tensor_head = torch.tensor(zero_tensor_head).to('cuda:0')

        e_time_dict = {}
        all_triples_array = np.array(all_triples.cpu())
        for item in all_triples_array:
        # for item in output_sample:
            if item[0] not in e_time_dict:
                e_time_dict[item[0]] = []
                e_time_dict[item[0]].append(item)
            else:
                e_time_dict[item[0]].append(item)
        time_list = []
        entity_list_total = []
        entity_history_index = {}
        for key in e_time_dict.keys():
            output_new = []
            output = []
            for i in range(len(e_time_dict[key])):
                output_new += [list(e_time_dict[key][i])]
            output_new = np.array(output_new)
            output_new.reshape(len(e_time_dict[key]), 3)
            output.append(output_new)
            entity_list = [key]
            entity_list_total.append(key)
            # 使用子图采样
            time_list_one = Get_time_list(t_to_entity, entity_list, time_idx+history_len, number)
            # 不使用子图采样
            # time_list_one = []
            # for i in range(time_idx + history_len-3, time_idx + history_len):
            #      time_list_one.append(i)
            if time_list_one == []:
                for i in range(number):
                    if time_idx+history_len-(i+1) not in time_list_one:
                        time_list_one.append(time_idx+history_len-(i+1))


            for time in time_list_one:
                if time not in time_list:
                    time_list.append(time)
            entity_history_index[key] = time_list_one
        entity_history_index_filtered = {key: value for key, value in entity_history_index.items() if value}
        time_list = sorted(time_list)
        history_index = {}
        for key in entity_history_index_filtered.keys():
            if key not in history_index.keys():
                history_index[int(key)] = []
            history_index[int(key)] = find_list_index(entity_history_index_filtered[key], np.array(time_list))
        # 使用子图采样
        input_list = Get_input_list(history_list, entity_list_total, time_list)
        if len(input_list) == number:
            flag = False
            for k in range(len(input_list)):
                if input_list[k].size == 0:
                    flag = True
            if flag:
                input_list = [history_list[i] for i in time_list]
        Dynic = True
        if input_list == []:
            input_list = history_list[time_idx+history_len - args.train_history_len:
                                        time_idx+history_len]
            Dynic = False
        # 不使用子图采样
        # input_list = [history_list[i] for i in time_list]
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
                                                                 test_triples_input, use_cuda, history_index, zero_tensor_head, total_entity, total_relation, entity_history_index_filtered, time_idx+history_len, number, Dynic)
        # test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
        #                                                          test_triples_input, use_cuda, history_index,
        #                                                          entity_history_index_filtered, time_idx + history_len)
        mrr_filter_snap_r, mrr_snap_r, rank_raw_r, rank_filter_r = utils.get_total_rank(test_triples, final_r_score,
                                                                                        all_ans_r_list[time_idx],
                                                                                        eval_bz=1000, rel_predict=1)
        mrr_filter_snap, mrr_snap, rank_raw, rank_filter = utils.get_total_rank(test_triples, final_score,
                                                                                all_ans_list[time_idx], eval_bz=1000,
                                                                                rel_predict=0)

        # used to global statistic
        ranks_raw.append(rank_raw)
        ranks_filter.append(rank_filter)
        # used to show slide results
        mrr_raw_list.append(mrr_snap)
        mrr_filter_list.append(mrr_filter_snap)

        # relation rank
        ranks_raw_r.append(rank_raw_r)
        ranks_filter_r.append(rank_filter_r)
        mrr_raw_list_r.append(mrr_snap_r)
        mrr_filter_list_r.append(mrr_filter_snap_r)

        history_list.append(test_snap)
        # reconstruct history graph list
        '''if args.multi_step:
            if not args.relation_evaluation:
                predicted_snap = utils.construct_snap(test_triples, num_nodes, num_rels, final_score, args.topk)
            else:
                predicted_snap = utils.construct_snap_r(test_triples, num_nodes, num_rels, final_r_score, args.topk)
            if len(predicted_snap):
                input_list.pop(0)
                input_list.append(predicted_snap)
        else:
            input_list.pop(0)
            input_list.append(test_snap)'''
        idx += 1

    mrr_raw = utils.stat_ranks(ranks_raw, "raw_ent", name, 'entity_pre_raw')
    # print(ranks_raw)
    mrr_filter = utils.stat_ranks(ranks_filter, "filter_ent", name, 'entity_pre_filter')
    mrr_raw_r = utils.stat_ranks(ranks_raw_r, "raw_rel", name, 'rel_pre_raw')
    mrr_filter_r = utils.stat_ranks(ranks_filter_r, "filter_rel", name, 'rel_pre_filter')
    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r

def Get_Time_to_Relation(train_list):
    t_to_relation = {}
    for i, data in enumerate(train_list):
        t_to_relation[i] = []
        for item in data:
            if item[1] not in t_to_relation[i]:
                t_to_relation[i].append(item[1])
    return t_to_relation

def Get_total_hr_to_tail(train_list, num_rels):
    hr_to_tail = {}
    for idx, data in enumerate(train_list):
        for item in data:
            if (item[0], item[1]) not in hr_to_tail:
                hr_to_tail[(item[0], item[1])] = []
                hr_to_tail[(item[0], item[1])].append((idx, item[2]))
            else:
                hr_to_tail[(item[0], item[1])].append((idx, item[2]))
            if (item[2], item[1] + num_rels) not in hr_to_tail:
                hr_to_tail[(item[2], item[1] + num_rels)] = []
                hr_to_tail[(item[2], item[1] + num_rels)].append((idx, item[0]))
            else:
                hr_to_tail[(item[2], item[1] + num_rels)].append((idx, item[0]))
    return hr_to_tail

def Get_Time_to_Entity(train_list):
    t_to_entity = {}
    for i, data in enumerate(train_list):
        t_to_entity[i] = []
        for item in data:
            if item[0] not in t_to_entity[i]:
                t_to_entity[i].append(item[0])
            if item[2] not in t_to_entity[i]:
                t_to_entity[i].append(item[2])
    return t_to_entity

def get_entity_time(train_data):
    e_to_time = {}
    for item in train_data:
        if item[0] not in e_to_time:
            e_to_time[item[0]] = []
            if item[3]-1 not in e_to_time[item[0]]:
                e_to_time[item[0]].append(item[3]-1)
        else:
            if item[3]-1 not in e_to_time[item[0]]:
                e_to_time[item[0]].append(item[3]-1)
    return e_to_time

def Get_input_list_new(train_list, relation_list, time_list):
    adj_list = []
    for time in time_list:
        one_list = []
        for item in train_list[time]:
            if item[1] in relation_list:
                one_list.append(item)
        one_list = np.array(one_list)
        # one_list.reshape(len(one_list), 3)
        adj_list.append(one_list)
    return adj_list

def Get_input_list(train_list, entity_list, time_list):
    adj_list = []
    for time in time_list:
        one_list = []
        for item in train_list[time]:
            if item[0] or item[2] in entity_list:
            # if item[0] in entity_list:
                one_list.append(item)
        one_list = np.array(one_list)
        # one_list.reshape(len(one_list), 3)
        adj_list.append(one_list)
    return adj_list

def Get_time_list_new(t_to_relation, relation_list, train_sample_num):
    t_num_relation = {}
    for time in range(train_sample_num):
        t_num_relation[time] = 0
        for entity in relation_list:
            if entity in t_to_relation[time]:
                t_num_relation[time] += 1
    t_num_relation = {key: value for key, value in t_num_relation.items() if value != 0}
    sorted_data = sorted(t_num_relation.items(), key=lambda x: x[0], reverse=True)
    top_3 = sorted_data[0:3]
    time_list = [item[0] for item in top_3]
    time_list = sorted(time_list)
    return time_list


def Get_time_list(t_to_entity, entity_list, train_sample_num, number):
    t_num_entity = {}
    for time in range(train_sample_num):
        t_num_entity[time] = 0
        for entity in entity_list:
            if entity in t_to_entity[time]:
                t_num_entity[time] += 1
    t_num_entity = {key: value for key, value in t_num_entity.items() if value != 0}
    sorted_data = sorted(t_num_entity.items(), key=lambda x: x[0], reverse=True)
    top_3 = sorted_data[0:number]
    time_list = [item[0] for item in top_3]
    time_list = sorted(time_list)
    return time_list

def find_three_consecutive_less_than(t_to_s, entity_sample, value):
    t_num_entity = {}
    for time in range(value):
        t_num_entity[time] = 0
        if entity_sample in t_to_s[time]:
            t_num_entity[time] += 1
    sorted_data = sorted(t_num_entity.items(), key=lambda x: x[1], reverse=True)
    top_3 = sorted_data[:3]
    time_list = [item[0] for item in top_3]
    return time_list

def find_list_index(list1,list2):
    mask = np.isin(list2, list1)
    return list(np.where(mask)[0])


def getTime_weight(unique_first_elements, train_sample_num):
    time_weight_dict = {}
    decay_factor = 0.1
    time_differences = np.abs(np.array(unique_first_elements) - train_sample_num)
    # 计算权重，使用指数递减函数
    weights = np.exp(-decay_factor * np.array(time_differences))
    # 归一化权重，使其和为1
    weights /= np.sum(weights)
    for i in range(len(unique_first_elements)):
        time_weight_dict[unique_first_elements[i]] = weights[i];
    return time_weight_dict



def get_entity_num(hr_tail, hr, train_sample_num, num_nodes):
    tail_entity = []
    total_entity_num = np.zeros(num_nodes)
    if hr in hr_tail.keys():
        time_tail = hr_tail[hr]
        unique_first_elements = list(set(t[0] for t in time_tail if t[0] < train_sample_num))
        # 对列表进行排序
        unique_first_elements.sort()
        time_weight_dict = getTime_weight(unique_first_elements, train_sample_num)
        for item in time_tail:
            if item[0] < train_sample_num:
                tail_entity.append((time_weight_dict[item[0]], item[1]))
        entity_num_dict = {}
        for entity in tail_entity:
            if entity[1] not in entity_num_dict:
                # entity_num_dict[entity] = 1
                entity_num_dict[entity[1]] = 0
                entity_num_dict[entity[1]] += 1*entity[0]
            else:
                # entity_num_dict[entity] = 1
                entity_num_dict[entity[1]] += 1*entity[0]
        for entity in entity_num_dict:
            total_entity_num[entity] = entity_num_dict[entity] #/ sum(entity_num_dict.values())
    return total_entity_num




def Get_head_relation(hr_tail, all_triples, train_sample_num, num_nodes):
    h_r_t = {}
    for idx, triplet in enumerate(all_triples):
        if (triplet[0].item(), triplet[1].item()) not in h_r_t:
            h_r_t[(triplet[0].item(), triplet[1].item())] = []
            h_r_t[(triplet[0].item(), triplet[1].item())].append((idx, get_entity_num(hr_tail, (triplet[0].item(), triplet[1].item()), train_sample_num, num_nodes)))
        else:
            h_r_t[(triplet[0].item(), triplet[1].item())].append((idx, get_entity_num(hr_tail, (triplet[0].item(), triplet[1].item()), train_sample_num, num_nodes)))

    return h_r_t

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    # set_random_seed(args.seed)

    # load graph data
    print("loading graph data")
    data = utils.load_data(args.dataset)
    train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    test_list = utils.split_by_time(data.test)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, False)
    all_ans_list_r_test = utils.load_all_answers_for_time_filter(data.test, num_rels, num_nodes, True)
    all_ans_list_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, False)
    all_ans_list_r_valid = utils.load_all_answers_for_time_filter(data.valid, num_rels, num_nodes, True)

    model_name = args.dataset
    model_state_file = '../models/' + model_name + "_number_2"
    print("Sanity Check: stat name : {}".format(model_state_file))
    print("Sanity Check: Is cuda available ? {}".format(torch.cuda.is_available()))

    use_cuda = args.gpu >= 0 and torch.cuda.is_available()

    if args.add_static_graph:
        static_triples = np.array(
            _read_triplets_as_list("../data/" + args.dataset + "/e-w-graph.txt", {}, {}, load_time=False))
        num_static_rels = len(np.unique(static_triples[:, 1]))
        num_words = len(np.unique(static_triples[:, 2]))
        static_triples[:, 2] = static_triples[:, 2] + num_nodes
        static_node_id = torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long().cuda(args.gpu) \
            if use_cuda else torch.from_numpy(np.arange(num_words + data.num_nodes)).view(-1, 1).long()
    else:
        num_static_rels, num_words, static_triples, static_graph = 0, 0, [], None

    # create stat
    model = RecurrentRGCN(args.decoder,
                          args.encoder,
                          num_nodes,
                          num_rels,
                          num_static_rels,
                          num_words,
                          args.n_hidden,
                          args.opn,
                          sequence_len=args.train_history_len,
                          num_bases=args.n_bases,
                          num_basis=args.n_basis,
                          num_hidden_layers=args.n_layers,
                          dropout=args.dropout,
                          self_loop=args.self_loop,
                          skip_connect=args.skip_connect,
                          layer_norm=args.layer_norm,
                          input_dropout=args.input_dropout,
                          hidden_dropout=args.hidden_dropout,
                          feat_dropout=args.feat_dropout,
                          aggregation=args.aggregation,
                          weight=args.weight,
                          discount=args.discount,
                          angle=args.angle,
                          use_static=args.add_static_graph,
                          entity_prediction=args.entity_prediction,
                          relation_prediction=args.relation_prediction,
                          use_cuda=use_cuda,
                          gpu=args.gpu,
                          analysis=args.run_analysis)

    if use_cuda:
        torch.cuda.set_device(args.gpu)
        model.cuda()

    if args.add_static_graph:
        static_graph = build_sub_graph(len(static_node_id), num_static_rels, static_triples, use_cuda, args.gpu)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.test and os.path.exists(model_state_file):
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(args.dataset,
                                                            model,
                                                            data,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            args.train_history_len,
                                                            "test")
    elif args.test and not os.path.exists(model_state_file):
        print("--------------{} not exist, Change mode to train and generate stat for testing----------------\n".format(
            model_state_file))
    else:
        print("----------------------------------------start training----------------------------------------\n")
        best_mrr = 0

        # mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(args.dataset,
        #                                                     model,
        #                                                     data,
        #                                                     train_list,
        #                                                     valid_list,
        #                                                     num_rels,
        #                                                     num_nodes,
        #                                                     use_cuda,
        #                                                     all_ans_list_valid,
        #                                                     all_ans_list_r_valid,
        #                                                     model_state_file,
        #                                                     static_graph,
        #                                                     args.train_history_len,
        #                                                     mode="train"
        #                                                     )
        hr_tail = Get_total_hr_to_tail(train_list, num_rels)
        t_to_entity = Get_Time_to_Entity(train_list)
        for epoch in range(args.n_epochs):
            model.train()
            losses = []
            losses_e = []
            losses_r = []
            losses_static = []

            idx = [_ for _ in range(len(train_list))]
            random.shuffle(idx)
            train_len = len(train_list)

            for train_sample_num in tqdm(idx):
                # train_sample_num = 2
                if train_sample_num == 0:
                    continue
                output = train_list[train_sample_num:train_sample_num + 1]
                # 模拟新出现实体
                output_new = output
                output_new = [torch.from_numpy(_).long().cuda() for _ in output_new] if use_cuda else [
                    torch.from_numpy(_).long() for _ in output_new]
                inverse_triples = output_new[0][:, [2, 1, 0]]
                inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
                all_triples = torch.cat([output_new[0], inverse_triples])
                total_entity = all_triples[:, 0]
                total_relation = all_triples[:, 1]
                h_r_t = Get_head_relation(hr_tail, all_triples, train_sample_num, num_nodes)
                zero_tensor_head = np.zeros((len(all_triples), num_nodes))
                for item in h_r_t:
                    for data in h_r_t[item]:
                        zero_tensor_head[data[0]] = data[1]

                zero_tensor_head[zero_tensor_head == 0] = -100
                zero_tensor_head = torch.tensor(zero_tensor_head).to(args.gpu)


                all_triples_array = np.array(all_triples.cpu())
                e_time_dict = {}
                for item in all_triples_array:
                # for item in output[0]:
                    if item[0] not in e_time_dict:
                        e_time_dict[item[0]] = []
                        e_time_dict[item[0]].append(item)
                    else:
                        e_time_dict[item[0]].append(item)
                time_list = []
                entity_list_total = []
                entity_history_index = {}
                for key in e_time_dict.keys():
                    if train_sample_num == 0: continue
                    entity_list = [key]
                    time_list_one = []
                    if train_sample_num - args.train_history_len < 0:
                        for time in range(train_sample_num):
                            time_list_one.append(time)
                    else:
                        # 不使用子图采样
                        # for i in range(train_sample_num - args.train_history_len, train_sample_num):
                        #     time_list_one.append(i)
                        # 使用子图采样
                        time_list_one = Get_time_list(t_to_entity, entity_list, train_sample_num, args.train_history_len)
                    entity_list_total.append(key)
                    for time in time_list_one:
                        if time not in time_list:
                            time_list.append(time)
                    entity_history_index[key] = time_list_one
                entity_history_index_filtered = {key: value for key, value in entity_history_index.items() if value}
                time_list = sorted(time_list)
                history_index = {}
                for key in entity_history_index_filtered.keys():
                    if key not in history_index.keys():
                        history_index[int(key)] = []
                    history_index[int(key)] = find_list_index(entity_history_index_filtered[key], np.array(time_list))

                input_list = Get_input_list(train_list, entity_list_total, time_list) #使用子图采样，公开数据集
                Dynic = True
                if input_list == []:
                    input_list = train_list[train_sample_num - args.train_history_len:
                                            train_sample_num]
                    Dynic = False
                # input_list = [train_list[i] for i in time_list] #不使用子图采样,对于我们的数据
                # generate history graph
                history_glist = [build_sub_graph(num_nodes, num_rels, snap, use_cuda, args.gpu) for snap in input_list]
                output = [torch.from_numpy(_).long().cuda() for _ in output] if use_cuda else [
                    torch.from_numpy(_).long() for _ in output]
                # loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda, history_index, entity_history_index_filtered, train_sample_num)

                loss_e, loss_r, loss_static = model.get_loss(history_glist, output[0], static_graph, use_cuda, history_index, zero_tensor_head, total_entity, total_relation, entity_history_index_filtered, train_sample_num, args.train_history_len, Dynic)
                loss = args.task_weight * loss_e + (1 - args.task_weight) * loss_r + loss_static

                losses.append(loss.item())
                losses_e.append(loss_e.item())
                losses_r.append(loss_r.item())
                losses_static.append(loss_static.item())

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
                optimizer.step()
                optimizer.zero_grad()

            print(
                "Epoch {:04d} | Ave Loss: {:.4f} | entity-relation-static:{:.4f}-{:.4f}-{:.4f} Best MRR {:.4f} | Model {} "
                .format(epoch, np.mean(losses), np.mean(losses_e), np.mean(losses_r), np.mean(losses_static), best_mrr,
                        model_name))

            # validation
            if epoch+1 and (epoch+1) % args.evaluate_every == 0:
                mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(args.dataset,
                                                                    model,
                                                                    data,
                                                                    train_list,
                                                                    valid_list,
                                                                    num_rels,
                                                                    num_nodes,
                                                                    use_cuda,
                                                                    all_ans_list_valid,
                                                                    all_ans_list_r_valid,
                                                                    model_state_file,
                                                                    static_graph,
                                                                    args.train_history_len,
                                                                    mode="train"
                                                                    )
                train_list = train_list[0:train_len]
                if not args.relation_evaluation:  # entity prediction evalution
                    if mrr_raw < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
                else:
                    if mrr_raw_r < best_mrr:
                        if epoch >= args.n_epochs:
                            break
                    else:
                        best_mrr = mrr_raw_r
                        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)
        mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r = test(args.dataset,
                                                            model,
                                                            data,
                                                            train_list + valid_list,
                                                            test_list,
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_test,
                                                            all_ans_list_r_test,
                                                            model_state_file,
                                                            static_graph,
                                                            args.train_history_len,
                                                            mode="test")

    return mrr_raw, mrr_filter, mrr_raw_r, mrr_filter_r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='REGCN')

    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--seed", type=int, default=10,
                        help="seed")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="batch-size")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--test", action='store_true', default=False,
                        help="load stat from dir and directly test")
    parser.add_argument("--run-analysis", action='store_true', default=False,
                        help="print log info")
    parser.add_argument("--run-statistic", action='store_true', default=False,
                        help="statistic the result")
    parser.add_argument("--multi-step", action='store_true', default=False,
                        help="do multi-steps inference without ground truth")
    parser.add_argument("--topk", type=int, default=10,
                        help="choose top k entities as results when do multi-steps without ground truth")
    parser.add_argument("--add-static-graph", action='store_true', default=False,
                        help="use the info of static graph")
    parser.add_argument("--add-rel-word", action='store_true', default=False,
                        help="use words in relaitons")
    parser.add_argument("--relation-evaluation", action='store_true', default=False,
                        help="save model accordding to the relation evalution")

    # configuration for encoder RGCN stat
    parser.add_argument("--weight", type=float, default=1,
                        help="weight of static constraint")
    parser.add_argument("--task-weight", type=float, default=0.7,
                        help="weight of entity prediction task")
    parser.add_argument("--discount", type=float, default=1,
                        help="discount of weight of static constraint")
    parser.add_argument("--angle", type=int, default=10,
                        help="evolution speed")

    parser.add_argument("--encoder", type=str, default="uvrgcn",
                        help="method of encoder")
    parser.add_argument("--aggregation", type=str, default="none",
                        help="method of aggregation")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--skip-connect", action='store_true', default=False,
                        help="whether to use skip connect in a RGCN Unit")
    parser.add_argument("--n-hidden", type=int, default=200,
                        help="number of hidden units")
    parser.add_argument("--opn", type=str, default="sub",
                        help="opn of compgcn")

    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-basis", type=int, default=100,
                        help="number of basis vector for compgcn")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--self-loop", action='store_true', default=True,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--layer-norm", action='store_true', default=False,
                        help="perform layer normalization in every layer of gcn ")
    parser.add_argument("--relation-prediction", action='store_true', default=False,
                        help="add relation prediction loss")
    parser.add_argument("--entity-prediction", action='store_true', default=False,
                        help="add entity prediction loss")
    parser.add_argument("--split_by_relation", action='store_true', default=False,
                        help="do relation prediction")

    # configuration for stat training
    parser.add_argument("--n-epochs", type=int, default=30,
                        help="number of minimum training epochs on each time step")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")

    # configuration for evaluating
    parser.add_argument("--evaluate-every", type=int, default=20,
                        help="perform evaluation every n epochs")

    # configuration for decoder
    parser.add_argument("--decoder", type=str, default="convtranse",
                        help="method of decoder")
    parser.add_argument("--input-dropout", type=float, default=0.2,
                        help="input dropout for decoder ")
    parser.add_argument("--hidden-dropout", type=float, default=0.2,
                        help="hidden dropout for decoder")
    parser.add_argument("--feat-dropout", type=float, default=0.2,
                        help="feat dropout for decoder")

    # configuration for sequences stat
    parser.add_argument("--train-history-len", type=int, default=10,
                        help="history length")
    parser.add_argument("--test-history-len", type=int, default=20,
                        help="history length for test")
    parser.add_argument("--dilate-len", type=int, default=1,
                        help="dilate history graph")

    # configuration for optimal parameters
    parser.add_argument("--grid-search", action='store_true', default=False,
                        help="perform grid search for best configuration")
    parser.add_argument("-tune", "--tune", type=str, default="n_hidden,n_layers,dropout,n_bases",
                        help="stat to use")
    parser.add_argument("--num-k", type=int, default=500,
                        help="number of triples generated")

    args = parser.parse_args()
    print(args)
    if args.grid_search:
        out_log = '{}.{}.gs'.format(args.dataset, args.encoder + "-" + args.decoder)
        o_f = open(out_log, 'w')
        print("** Grid Search **")
        o_f.write("** Grid Search **\n")
        hyperparameters = args.tune.split(',')

        if args.tune == '' or len(hyperparameters) < 1:
            print("No hyperparameter specified.")
            sys.exit(0)
        grid = hp_range[hyperparameters[0]]
        for hp in hyperparameters[1:]:
            grid = itertools.product(grid, hp_range[hp])
        hits_at_1s = {}
        hits_at_10s = {}
        mrrs = {}
        grid = list(grid)
        print('* {} hyperparameter combinations to try'.format(len(grid)))
        o_f.write('* {} hyperparameter combinations to try\n'.format(len(grid)))
        o_f.close()

        for i, grid_entry in enumerate(list(grid)):
            o_f = open(out_log, 'a')
            if not (type(grid_entry) is list or type(grid_entry) is list):
                grid_entry = [grid_entry]
            grid_entry = utils.flatten(grid_entry)
            print('* Hyperparameter Set {}:'.format(i))
            o_f.write('* Hyperparameter Set {}:\n'.format(i))
            signature = ''
            print(grid_entry)
            o_f.write("\t".join([str(_) for _ in grid_entry]) + "\n")
            # def run_experiment(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
            mrr, hits, ranks = run_experiment(args, grid_entry[0], grid_entry[1], grid_entry[2], grid_entry[3])
            print("MRR (raw): {:.6f}".format(mrr))
            o_f.write("MRR (raw): {:.6f}\n".format(mrr))
            for hit in hits:
                avg_count = torch.mean((ranks <= hit).float())
                print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
                o_f.write("Hits (raw) @ {}: {:.6f}\n".format(hit, avg_count.item()))
    # single run
    else:
        run_experiment(args)
    sys.exit()



