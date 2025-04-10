from rgcn import utils
import argparse
import os
import sys
import numpy as np
import torch
from rgcn.knowledge_graph import _read_triplets_as_list
from src.rrgcn import RecurrentRGCN
from rgcn.utils import build_sub_graph
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.table import Table
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

def get_entity_num(hr_tail, hr, train_sample_num, num_nodes):
    tail_entity = []
    total_entity_num = np.zeros(num_nodes)
    if hr in hr_tail.keys():
        time_tail = hr_tail[hr]
        for item in time_tail:
            if item[0] < train_sample_num:
                tail_entity.append(item[1])
        entity_num_dict = {}
        for entity in tail_entity:
            if entity not in entity_num_dict:
                # entity_num_dict[entity] = 1
                entity_num_dict[entity] = 0
                entity_num_dict[entity] += 1
            else:
                # entity_num_dict[entity] = 1
                entity_num_dict[entity] += 1
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

def find_list_index(list1,list2):
    mask = np.isin(list2, list1)
    return list(np.where(mask)[0])

def Get_time_list(t_to_entity, entity_list, train_sample_num):
    t_num_entity = {}
    for time in range(train_sample_num):
        t_num_entity[time] = 0
        for entity in entity_list:
            if entity in t_to_entity[time]:
                t_num_entity[time] += 1
    t_num_entity = {key: value for key, value in t_num_entity.items() if value != 0}
    sorted_data = sorted(t_num_entity.items(), key=lambda x: x[0], reverse=True)
    top_3 = sorted_data[0:3]
    time_list = [item[0] for item in top_3]
    time_list = sorted(time_list)
    return time_list

def Get_input_list(train_list, entity_list, time_list):
    adj_list = []
    for time in time_list:
        one_list = []
        for item in train_list[time]:
            if item[0] in entity_list or item[2] in entity_list:
                one_list.append(item)
        one_list = np.array(one_list)
        # one_list.reshape(len(one_list), 3)
        adj_list.append(one_list)
        # if one_list != []:
        #     one_list = np.array(one_list)
        #     # one_list.reshape(len(one_list), 3)
        #     adj_list.append(one_list)
        # else:
        #     one_list = train_list[time]
        #     adj_list.append(one_list)
    return adj_list

def test(name, model, data, history_list, test_list, num_rels, num_nodes, use_cuda, all_ans_list, all_ans_r_list, model_name,
         static_graph, mode):
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
        # test_snap = test_list[2]
        output_sample = test_snap
        test_snap = np.array([[1569, 13, 1570]])
        # 模拟新出现实体
        output_new = test_snap
        output_new = torch.LongTensor(output_new).cuda() if use_cuda else torch.LongTensor(output_new)
        # 注释
        # inverse_triples = output_new[:, [2, 1, 0]]
        # inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
        # all_triples = torch.cat([output_new, inverse_triples])
        # total_entity = all_triples[:, 0]
        # total_relation = all_triples[:, 1]
        # 注释结束，测试一个
        all_triples = output_new
        total_entity = all_triples[:, 0]
        total_relation = all_triples[:, 1]
        # 测试一个结束
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
            time_list_one = Get_time_list(t_to_entity, entity_list, time_idx+history_len)
            # 不使用子图采样
            # time_list_one = []
            # for i in range(time_idx + history_len-3, time_idx + history_len):
            #      time_list_one.append(i)
            if time_list_one == []:
                for i in range(3):
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

        # 不使用子图采样
        # input_list = [history_list[i] for i in time_list]
        history_glist = [build_sub_graph(num_nodes, num_rels, g, use_cuda, args.gpu) for g in input_list]
        test_triples_input = torch.LongTensor(test_snap).cuda() if use_cuda else torch.LongTensor(test_snap)
        test_triples_input = test_triples_input.to(args.gpu)
        test_triples, final_score, final_r_score = model.predict(history_glist, num_rels, static_graph,
                                                                 test_triples_input, use_cuda, history_index, zero_tensor_head, total_entity, total_relation, entity_history_index_filtered, time_idx+history_len)
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


def fact_num(sample, search_list):
    num = 0
    for i in range(len(search_list)):
        i_num = 0
        for fact in search_list[i]:
            if sample[0] == fact[0] or sample[0] == fact[2]:
            # if np.array_equal(sample, fact):
                i_num += 1
        if i_num != 0:
            num += 1
    return num


def data_split_num(train_list, valid_list):
    occur_num_dict = {}
    occur_num_dict[0] = []
    occur_num_dict[1] = []
    # occur_num_dict[2] = []
    # occur_num_dict[3] = []
    # occur_num_dict[4] = []
    # occur_num_dict[5] = []
    # occur_num_dict[6] = []
    # occur_num_dict[7] = []
    # occur_num_dict[8] = []
    # occur_num_dict[9] = []
    # occur_num_dict[10] = []
    for i in range(len(valid_list)):
        for fact in valid_list[i]:
            num_fact = fact_num(fact, train_list[-1:])
            if num_fact == 0:
                occur_num_dict[0].append(list(fact)+[i])
            if num_fact == 1:
                occur_num_dict[1].append(list(fact)+[i])
            # if num_fact == 2:
            #     occur_num_dict[2].append(list(fact)+[i])
            # if num_fact == 3:
            #     occur_num_dict[3].append(list(fact)+[i])
            # if num_fact == 4:
            #     occur_num_dict[4].append(list(fact) + [i])
            # if num_fact == 5:
            #     occur_num_dict[5].append(list(fact) + [i])
            # if num_fact == 6:
            #     occur_num_dict[6].append(list(fact) + [i])
            # if num_fact == 7:
            #     occur_num_dict[7].append(list(fact) + [i])
            # if num_fact == 8:
            #     occur_num_dict[8].append(list(fact) + [i])
            # if num_fact == 9:
            #     occur_num_dict[9].append(list(fact) + [i])
            # if num_fact == 10:
            #     occur_num_dict[10].append(list(fact) + [i])
        train_list.append(valid_list[i])
    occur_num_dict_new = {}
    for num in occur_num_dict:
        occur_num_dict_new[num] = {}
        for item in occur_num_dict[num]:
            if item[-1] not in occur_num_dict_new[num]:
                occur_num_dict_new[num][item[-1]] = []
                occur_num_dict_new[num][item[-1]].append(item[:3])
            else:
                occur_num_dict_new[num][item[-1]].append(item[:3])
    occur_num_dict_new_new = {}
    for num in range(len(occur_num_dict_new)-1):
        occur_num_dict_new_new[num] = []
        for time in range(len(occur_num_dict_new[num])):
            occur_num_dict_new_new[num].append(np.array(occur_num_dict_new[num][time]))

    return occur_num_dict_new_new


def run_data_split(args, n_hidden=None, n_layers=None, dropout=None, n_bases=None):
    # load configuration for grid search the best configuration
    if n_hidden:
        args.n_hidden = n_hidden
    if n_layers:
        args.n_layers = n_layers
    if dropout:
        args.dropout = dropout
    if n_bases:
        args.n_bases = n_bases

    print("loading graph data")
    data = utils.load_data(args.dataset)
    test_list = utils.split_by_time(data.test)
    train_list = utils.split_by_time(data.train)
    true_train_list = utils.split_by_time(data.train)
    valid_list = utils.split_by_time(data.valid)
    occur_num_dict_new = data_split_num(train_list+valid_list, test_list)

    num_nodes = data.num_nodes
    num_rels = data.num_rels

    all_ans_list_valid_0 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[0], num_rels, num_nodes, False)
    all_ans_list_r_valid_0 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[0], num_rels, num_nodes, True)

    all_ans_list_valid_1 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[1], num_rels, num_nodes, False)
    all_ans_list_r_valid_1 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[1], num_rels, num_nodes, True)

    all_ans_list_valid_2 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[2], num_rels, num_nodes, False)
    all_ans_list_r_valid_2 = utils.load_all_answers_for_time_filter_new(occur_num_dict_new[2], num_rels, num_nodes, True)

    # 测试
    model_name = args.dataset
    model_state_file = '../models/' + model_name
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
                                                            true_train_list,
                                                            occur_num_dict_new[0],
                                                            num_rels,
                                                            num_nodes,
                                                            use_cuda,
                                                            all_ans_list_valid_0,
                                                            all_ans_list_r_valid_0,
                                                            model_state_file,
                                                            static_graph,
                                                            "test")

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
    parser.add_argument("--n-epochs", type=int, default=20,
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
    run_data_split(args)
    sys.exit()
