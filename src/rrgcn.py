import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import dgl
# from rgcn.layers import RGCNBlockLayer as RGCNLayer
from ..rgcn.layers import UnionRGCNLayer, RGCNBlockLayer
from ..src.model import BaseRGCN
from ..src.decoder import ConvTransE, ConvTransR
from transformers import BertModel, BertTokenizer
# import matplotlib.pyplot as plt
class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        print("activate function: {}".format(act))
        if self.skip_connect:
            sc = False if idx == 0 else True
        else:
            sc = False
        if self.encoder_name == "uvrgcn":
            return UnionRGCNLayer(self.h_dim, self.h_dim, self.num_rels, self.num_bases,
                             activation=act, dropout=self.dropout, self_loop=self.self_loop, skip_connect=sc, rel_emb=self.rel_emb)
        else:
            raise NotImplementedError


    def forward(self, g, init_ent_emb, init_rel_emb):
        if self.encoder_name == "uvrgcn":
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            x, r = init_ent_emb, init_rel_emb
            for i, layer in enumerate(self.layers):
                layer(g, [], r[i])
            return g
        else:
            if self.features is not None:
                print("----------------Feature is not None, Attention ------------")
                g.ndata['id'] = self.features
            node_id = g.ndata['id'].squeeze()
            g.ndata['h'] = init_ent_emb[node_id]
            if self.skip_connect:
                prev_h = []
                for layer in self.layers:
                    prev_h = layer(g, prev_h)
            else:
                for layer in self.layers:
                    layer(g, [])
            return g.ndata.pop('h')

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class SelfAttention(nn.Module):
    def __init__(self, h_dim):
        super(SelfAttention, self).__init__()
        self.W_q = nn.Linear(h_dim, h_dim)
        self.W_k = nn.Linear(h_dim, h_dim)  # 使用当前时刻和上一时刻的隐藏状态生成 K
        self.W_v = nn.Linear(h_dim, h_dim)
        # self.W_k = nn.Linear(2 * h_dim, h_dim)  # 使用当前时刻和上一时刻的隐藏状态生成 K
        # self.W_v = nn.Linear(2 * h_dim, h_dim)

    def forward(self, current_h, prev_h):
        # Calculate query, key, and value vectors
        q = self.W_q(current_h)
        k = self.W_k(prev_h)  # 拼接当前时刻和上一时刻的隐藏状态
        v = self.W_v(prev_h)
        # k = self.W_k(torch.cat((current_h, prev_h), dim=-1))  # 拼接当前时刻和上一时刻的隐藏状态
        # v = self.W_v(torch.cat((current_h, prev_h), dim=-1))
        # Calculate attention scores using dot product
        attention_scores = torch.matmul(q, k.transpose(0, 1)) / torch.sqrt(
            torch.tensor(q.size(-1), dtype=torch.float32))

        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Use attention weights to compute the weighted sum of values
        attention_output = torch.matmul(attention_weights, v)

        # Combine the current hidden state with the attention output
        # new_h = prev_h + attention_output
        att_weight = F.sigmoid(attention_output)

        return att_weight

# 定义Transformer编码器层
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        # src2, _ = self.self_attn(src, src, src, attn_mask=mask)
        src2, _ = self.self_attn(src.transpose(1, 0), src.transpose(1, 0), src.transpose(1, 0), attn_mask=mask)
        src = src + self.dropout(src2.transpose(1, 0))
        src = self.norm1(src)
        src2 = self.feed_forward(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        # if i == 5:
        #     view_maxtri = _.cpu().numpy()
        #     view_maxtri = view_maxtri + 0.5
        #     time_steps = [131, 132, 133, 140]
        #     time_steps2 = [140, 133, 132, 131]
        #     # 创建一个热力图
        #     plt.imshow(view_maxtri, cmap='seismic', interpolation='nearest')
        #     # 添加颜色条
        #     plt.colorbar()
        #     # 添加横坐标和纵坐标的标签
        #     plt.xticks(np.arange(len(time_steps)), time_steps, fontsize=16)
        #     plt.yticks(np.arange(len(time_steps)), time_steps2, fontsize=16)
        #     # 显示网格
        #     plt.grid(visible=True, linestyle='--', linewidth=0.5, alpha=0.5)
        #     # 显示图像
        #     plt.show()
        return src

# 定义完整的Transformer编码器模型
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])

    def forward(self, src, mask=None):
        # i = 0
        for layer in self.layers:
            # src = layer(i, src, mask)
            src = layer(src, mask)
            # i += 1
        return src

class RecurrentRGCN(nn.Module):
    def __init__(self, decoder_name, encoder_name, num_ents, num_rels, num_static_rels, num_words, h_dim, opn, sequence_len, num_bases=-1, num_basis=-1,
                 num_hidden_layers=1, dropout=0, self_loop=False, skip_connect=False, layer_norm=False, input_dropout=0,
                 hidden_dropout=0, feat_dropout=0, aggregation='cat', weight=1, discount=0, angle=0, use_static=False,
                 entity_prediction=False, relation_prediction=False, use_cuda=False,
                 gpu = 0, analysis=False):
        super(RecurrentRGCN, self).__init__()

        self.decoder_name = decoder_name
        self.encoder_name = encoder_name
        self.num_rels = num_rels
        self.num_ents = num_ents
        self.opn = opn
        self.num_words = num_words
        self.num_static_rels = num_static_rels
        self.sequence_len = sequence_len
        self.h_dim = h_dim
        self.layer_norm = layer_norm
        self.h = None
        self.run_analysis = analysis
        self.aggregation = aggregation
        self.relation_evolve = False
        self.weight = weight
        self.discount = discount
        self.use_static = use_static
        self.angle = angle
        self.relation_prediction = relation_prediction
        self.entity_prediction = entity_prediction
        self.emb_rel = None
        self.gpu = gpu

        self.w1 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w1)

        self.w2 = torch.nn.Parameter(torch.Tensor(self.h_dim, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.w2)

        self.emb_rel = torch.nn.Parameter(torch.Tensor(self.num_rels * 2, self.h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.emb_rel)

        self.dynamic_emb = torch.nn.Parameter(torch.Tensor(num_ents, h_dim), requires_grad=True).float()
        torch.nn.init.normal_(self.dynamic_emb)

        if self.use_static:
            self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, h_dim), requires_grad=True).float()
            torch.nn.init.xavier_normal_(self.words_emb)
            self.statci_rgcn_layer = RGCNBlockLayer(self.h_dim, self.h_dim, self.num_static_rels*2, num_bases,
                                                    activation=F.rrelu, dropout=dropout, self_loop=False, skip_connect=False)
            self.static_loss = torch.nn.MSELoss()

        self.loss_r = torch.nn.CrossEntropyLoss()
        self.loss_e = torch.nn.CrossEntropyLoss()

        self.rgcn = RGCNCell(num_ents,
                             h_dim,
                             h_dim,
                             num_rels * 2,
                             num_bases,
                             num_basis,
                             num_hidden_layers,
                             dropout,
                             self_loop,
                             skip_connect,
                             encoder_name,
                             self.opn,
                             self.emb_rel,
                             use_cuda,
                             analysis)

        self.time_gate_weight = nn.Parameter(torch.Tensor(h_dim, h_dim))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(h_dim))
        nn.init.zeros_(self.time_gate_bias)

        self.cos = torch.cos
        self.time_weight = torch.nn.Parameter(torch.Tensor(1, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.time_weight)
        self.time_bias = torch.nn.Parameter(torch.Tensor(1, h_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.time_bias)

        self.mlp_weight = torch.nn.Parameter(torch.Tensor(h_dim * 2, self.num_ents))
        torch.nn.init.xavier_uniform_(self.mlp_weight, gain=nn.init.calculate_gain('relu'))
        self.mlp_bias = torch.nn.Parameter(torch.Tensor(self.num_ents))
        nn.init.zeros_(self.mlp_bias)

        # self.rnn_hidden_dim = h_dim
        # self.rnn = nn.RNN(input_size=h_dim * 2, hidden_size=self.rnn_hidden_dim, batch_first=True)

        # 注意力机制
        # self.attention = SelfAttention(self.h_dim)

        # GRU cell for entity evolving
        # self.entity_cell_1 = nn.GRUCell(self.h_dim, self.h_dim)

        # GRU cell for relation evolving
        # self.relation_cell_1 = nn.GRUCell(self.h_dim*2, self.h_dim)

        #transformer encoder
        # 使用预训练的BERT模型和tokenizer
        self.transformer_encoder = TransformerEncoder(6, 200, 8, 200, 0.1)


        # decoder
        if decoder_name == "convtranse":
            self.decoder_ob = ConvTransE(num_ents, h_dim, input_dropout, hidden_dropout, feat_dropout)
            self.rdecoder = ConvTransR(num_rels, h_dim, input_dropout, hidden_dropout, feat_dropout)
        else:
            raise NotImplementedError

    def get_time_embedding(self, cur_ts, history_t):
        time_interval = (cur_ts-history_t).cuda()
        time_interval = time_interval.unsqueeze(1).float()
        t = self.cos(time_interval*self.time_weight+self.time_bias)
        return t

    # def forward(self, g_list, static_graph, use_cuda, history_index, total_entity, total_relation,
    #             entity_history_index_filtered, train_sample_num, number):
    #     # def forward(self, g_list, static_graph, use_cuda, history_index, entity_history_index_filtered, train_sample_num):
    #
    #     gate_list = []
    #     degree_list = []
    #
    #     if self.use_static:
    #         static_graph = static_graph.to(self.gpu)
    #         static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
    #         self.statci_rgcn_layer(static_graph, [])
    #         static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
    #         static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
    #         self.h = static_emb
    #     else:
    #         self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
    #         static_emb = None
    #
    #     # 新加方法
    #     history_embs = []
    #     calculate_embs = []
    #     history_embs2 = []  # 不使用子图采样时
    #     self.init_r = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel
    #     self.init_h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
    #     self.gruh = self.h
    #     local_h = self.h.clone()
    #
    #     if len(g_list) == 0:
    #         history_embs.append(self.h)
    #         calculate_embs.append(self.h)
    #
    #     batched_g = dgl.batch(g_list)  # 把所有小图打包为一个大图
    #     batch_num_nodes = batched_g.batch_num_nodes()
    #     batch_num_edges = batched_g.batch_num_edges()
    #     batched_g = batched_g.to(self.gpu)
    #
    #     # batched_glist = []
    #     # for num in range(len(g_list)):
    #     #     batched_g = self.rgcn.forward(g_list[num], self.gruh, [self.init_r, self.init_r])
    #     #     batched_glist.append(batched_g)
    #
    #     batched_g = self.rgcn.forward(batched_g, self.gruh, [self.init_r,
    #                                                          self.init_r,
    #                                                          self.init_r])  # 相当于说是g_list中的每个g是聚合了4次num_layers*2；如果使用全局的话,self.gruh是全局的实体的初始化
    #     batched_glist = dgl.unbatch(batched_g, batch_num_nodes, batch_num_edges)
    #
    #     # 不使用子图采样时
    #     # for i in range(len(batched_glist)):
    #     #     history_embs2.append(batched_glist[i].ndata['h'])
    #
    #     # 使用子图采样
    #     non_zero = [key for key in history_index.keys() if len(history_index[key]) != 0]
    #     entity_history_emb = []
    #     entity_history_dgl_tensor = None
    #     for history_idex in range(0, number):
    #         entity_history_emb.append(local_h.clone())
    #     for key in non_zero:
    #         for i in history_index[key]:
    #             if entity_history_dgl_tensor is None:
    #                 entity_history_dgl_tensor = batched_glist[i].ndata['h'][key].unsqueeze(0)
    #             else:
    #                 temp = batched_glist[i].ndata['h'][key].unsqueeze(0)
    #
    #                 entity_history_dgl_tensor = torch.cat((entity_history_dgl_tensor, temp), 0)
    #         for buzu in range(0, number - len(history_index[key])):
    #             entity_history_dgl_tensor = torch.cat(
    #                 (entity_history_dgl_tensor, entity_history_dgl_tensor[-1].unsqueeze(0)))
    #     if entity_history_dgl_tensor is not None:
    #         assert (len(non_zero) * number == entity_history_dgl_tensor.size()[0])
    #
    #     # assert (len(non_zero) * number == entity_history_dgl_tensor.size()[0])
    #
    #     # for idx, history_emb in enumerate(entity_history_emb):  # 这步是提升的关键，进行赋值
    #     #      history_emb[non_zero] = entity_history_dgl_tensor[idx::3]
    #
    #     used_cal = self.h.clone()
    #
    #     # stacked_tensor = torch.stack(history_embs2, dim=1) #不使用子图采样
    #     # # stacked_tensor = torch.stack(entity_history_emb, dim=1) #使用子图采样
    #     # used_cal = used_cal.reshape(self.num_ents, 1, 200)
    #     # stacked_tensor = torch.cat((used_cal, stacked_tensor), dim=1)
    #     # reshaped_tensor = stacked_tensor.view(self.num_ents, -1)
    #     #
    #     # time_interval = [i for i in range(len(history_embs2) + 1)]  # 不使用子图采样
    #     #
    #     # history_glist_np = torch.from_numpy(np.array(time_interval)).cuda() if use_cuda else torch.from_numpy(
    #     #     np.array(time_interval))
    #     # time_emb = self.get_time_embedding(4, history_glist_np)
    #     # repeated_tensor_time = time_emb.repeat(self.num_ents, 1, 1)
    #     #
    #     # # 添加位置嵌入
    #     # concatenated_embeddings = reshaped_tensor.view(self.num_ents, -1, 200)
    #     # concatenated_embeddings = concatenated_embeddings + repeated_tensor_time
    #     #
    #     # batch_size = 128
    #     # total_samples = concatenated_embeddings.shape[0]
    #     # num_batches = (total_samples + batch_size - 1) // batch_size
    #     # batched_tensors = []
    #     # for i in range(num_batches):
    #     #     start_idx = i * batch_size
    #     #     end_idx = min((i + 1) * batch_size, total_samples)
    #     #     batched_tensor = concatenated_embeddings[start_idx:end_idx, :, :]
    #     #     batched_tensors.append(batched_tensor)
    #     # sample_total_tensors = []
    #     # for i in range(len(batched_tensors)):
    #     #     output = self.transformer_encoder(batched_tensors[i])
    #     #     # sample_total_tensors.append(output[:, 3, :])
    #     #     sample_total_tensors.append(output)
    #     #
    #     # merged_tensor = torch.cat(sample_total_tensors, dim=0)
    #     # # history_embs.append(merged_tensor)
    #     # # 增加时间门
    #     # for i in range(merged_tensor.size()[1]-1):
    #     #     current_h = merged_tensor[:, i+1, :].squeeze(-1)
    #     #     used_cal = F.normalize(current_h) if self.layer_norm else current_h
    #     #     time_weight = F.sigmoid(torch.mm(used_cal, self.time_gate_weight) + self.time_gate_bias)
    #     #     used_cal = time_weight * used_cal + (1 - time_weight) * self.gruh
    #     #     history_embs.append(used_cal)
    #     #     calculate_embs.append(used_cal)
    #     #
    #     # total_relation_embs = self.init_r[total_relation]
    #     # pre_emb = F.normalize(history_embs[-1]) if self.layer_norm else history_embs[-1]
    #     # total_entity_embs = pre_emb[total_entity]
    #     # concatenated_embedding = torch.cat((total_entity_embs, total_relation_embs), dim=1)
    #     # query_embedding = torch.mm(concatenated_embedding, self.mlp_weight) + self.mlp_bias
    #     # query_embedding = torch.tanh(query_embedding)
    #     # 结束
    #
    #     # 新新方法，每个实体分开送入transformer,然后再赋值,使用transformer
    #     entity_history_dgl_tensor = entity_history_dgl_tensor.view(-1, number, 200)
    #     entity_first_one = used_cal[non_zero].view(-1, 1, 200)
    #     cat_entity = torch.cat((entity_first_one, entity_history_dgl_tensor), dim=1)
    #     repeated_tensor_time = []
    #     for key in entity_history_index_filtered:
    #         entity_history_index_filtered[key].append(entity_history_index_filtered[key][0] - 1)
    #         entity_history_index_filtered[key] = sorted(entity_history_index_filtered[key])
    #         if len(entity_history_index_filtered[key]) != number + 1:
    #             for j in range(number + 1 - len(entity_history_index_filtered[key])):
    #                 entity_history_index_filtered[key].append(entity_history_index_filtered[key][-1])
    #         entity_history_index_filtered[key] = sorted(entity_history_index_filtered[key])
    #         entity_history_index_filtered[key] = torch.from_numpy(
    #             np.array(entity_history_index_filtered[key])).cuda() if use_cuda else torch.from_numpy(
    #             np.array(entity_history_index_filtered[key]))
    #         time_emb = self.get_time_embedding(train_sample_num, entity_history_index_filtered[key])
    #         repeated_tensor_time.append(time_emb)
    #
    #     repeated_tensor_time = torch.stack(repeated_tensor_time, dim=0)
    #
    #     finally_tensor = cat_entity + repeated_tensor_time
    #     # 划分batch
    #     batch_size = 128
    #     total_samples = finally_tensor.shape[0]
    #     num_batches = (total_samples + batch_size - 1) // batch_size
    #     batched_tensors = []
    #     for i in range(num_batches):
    #         start_idx = i * batch_size
    #         end_idx = min((i + 1) * batch_size, total_samples)
    #         batched_tensor = finally_tensor[start_idx:end_idx, :, :]
    #         batched_tensors.append(batched_tensor)
    #     sample_total_tensors = []
    #     for i in range(len(batched_tensors)):
    #         output = self.transformer_encoder(batched_tensors[i])
    #         sample_total_tensors.append(output)
    #
    #     output = torch.cat(sample_total_tensors, dim=0)
    #     # output = self.transformer_encoder(finally_tensor)
    #     # used_cal[non_zero] = output[:, 3, :]
    #     output = output[:, 1:number + 1, :]
    #     output = output.reshape(output.size()[0] * number, 200)
    #     for idx, history_emb in enumerate(entity_history_emb):  # 这步是提升的关键，进行赋值
    #         history_emb[non_zero] = output[idx::number]
    #     for current_g in entity_history_emb:
    #         current_h = F.normalize(current_g) if self.layer_norm else current_g
    #         time_weight = F.sigmoid(torch.mm(current_h, self.time_gate_weight) + self.time_gate_bias)
    #         used_cal = time_weight * current_h + (1 - time_weight) * self.gruh
    #         history_embs.append(used_cal)
    #         calculate_embs.append(used_cal)
    #     # 新结束
    #
    #     # 不使用transformer
    #     # for idx, history_emb in enumerate(entity_history_emb):  # 这步是提升的关键，进行赋值
    #     #     history_emb[non_zero] = entity_history_dgl_tensor[idx::3]
    #     #
    #     # # current_h = torch.zeros((entity_history_emb[0].size()[0], entity_history_emb[0].size()[1])).cuda().to(self.gpu)
    #     # # for current_g in entity_history_emb:
    #     # #     current_h += F.normalize(current_g) if self.layer_norm else current_g
    #     # current_h = entity_history_emb[-1]
    #     # time_weight = F.sigmoid(torch.mm(current_h, self.time_gate_weight) + self.time_gate_bias)
    #     # used_cal = time_weight * current_h + (1 - time_weight) * self.gruh
    #     # history_embs.append(used_cal)
    #     # calculate_embs.append(used_cal)
    #
    #     total_relation_embs = self.init_r[total_relation]
    #     pre_emb = F.normalize(history_embs[-1]) if self.layer_norm else history_embs[-1]
    #     total_entity_embs = pre_emb[total_entity]
    #     concatenated_embedding = torch.cat((total_entity_embs, total_relation_embs), dim=1)
    #     query_embedding = torch.mm(concatenated_embedding, self.mlp_weight) + self.mlp_bias
    #     query_embedding = torch.tanh(query_embedding)
    #
    #     # return history_embs, static_emb, self.init_r, gate_list, degree_list
    #     return history_embs, static_emb, self.init_r, gate_list, degree_list, query_embedding

    def forward(self, g_list, static_graph, use_cuda, history_index, total_entity, total_relation,
                entity_history_index_filtered, train_sample_num, number, Dynic):
        gate_list = []
        degree_list = []

        if self.use_static:
            static_graph = static_graph.to(self.gpu)
            static_graph.ndata['h'] = torch.cat((self.dynamic_emb, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.statci_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_ents, :]
            static_emb = F.normalize(static_emb) if self.layer_norm else static_emb
            self.h = static_emb
        else:
            self.h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
            static_emb = None

        # 新加方法
        history_embs = []
        calculate_embs = []
        history_embs2 = [] # 不使用子图采样时
        self.init_r = F.normalize(self.emb_rel) if self.layer_norm else self.emb_rel
        self.init_h = F.normalize(self.dynamic_emb) if self.layer_norm else self.dynamic_emb[:, :]
        self.gruh = self.h
        local_h = self.h.clone()

        if len(g_list) == 0:
            history_embs.append(self.h)
            calculate_embs.append(self.h)

        batched_g = dgl.batch(g_list)  # 把所有小图打包为一个大图
        batch_num_nodes = batched_g.batch_num_nodes()
        batch_num_edges = batched_g.batch_num_edges()
        batched_g = batched_g.to(self.gpu)


        batched_g = self.rgcn.forward(batched_g, self.gruh, [self.init_r,
                                                             self.init_r,
                                                             self.init_r])  # 相当于说是g_list中的每个g是聚合了4次num_layers*2；如果使用全局的话,self.gruh是全局的实体的初始化
        batched_glist = dgl.unbatch(batched_g, batch_num_nodes, batch_num_edges)

        if Dynic is False:
            # 不使用子图采样时
            for i in range(len(batched_glist)):
                history_embs2.append(batched_glist[i].ndata['h'])

            used_cal = self.h.clone()
            stacked_tensor = torch.stack(history_embs2, dim=1) #不使用子图采样
            # stacked_tensor = torch.stack(entity_history_emb, dim=1) #使用子图采样
            used_cal = used_cal.reshape(self.num_ents, 1, 200)
            stacked_tensor = torch.cat((used_cal, stacked_tensor), dim=1)
            reshaped_tensor = stacked_tensor.view(self.num_ents, -1)

            time_interval = [i for i in range(len(history_embs2) + 1)]  # 不使用子图采样

            history_glist_np = torch.from_numpy(np.array(time_interval)).cuda() if use_cuda else torch.from_numpy(
                np.array(time_interval))
            time_emb = self.get_time_embedding(4, history_glist_np)
            repeated_tensor_time = time_emb.repeat(self.num_ents, 1, 1)

            # 添加位置嵌入
            concatenated_embeddings = reshaped_tensor.view(self.num_ents, -1, 200)
            concatenated_embeddings = concatenated_embeddings + repeated_tensor_time

            batch_size = 128
            total_samples = concatenated_embeddings.shape[0]
            num_batches = (total_samples + batch_size - 1) // batch_size
            batched_tensors = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                batched_tensor = concatenated_embeddings[start_idx:end_idx, :, :]
                batched_tensors.append(batched_tensor)
            sample_total_tensors = []
            for i in range(len(batched_tensors)):
                output = self.transformer_encoder(batched_tensors[i])
                # sample_total_tensors.append(output[:, 3, :])
                sample_total_tensors.append(output)

            merged_tensor = torch.cat(sample_total_tensors, dim=0)
            # history_embs.append(merged_tensor)
            # 增加时间门
            for i in range(merged_tensor.size()[1]-1):
                current_h = merged_tensor[:, i+1, :].squeeze(-1)
                used_cal = F.normalize(current_h) if self.layer_norm else current_h
                time_weight = F.sigmoid(torch.mm(used_cal, self.time_gate_weight) + self.time_gate_bias)
                used_cal = time_weight * used_cal + (1 - time_weight) * self.gruh
                history_embs.append(used_cal)
                calculate_embs.append(used_cal)

            total_relation_embs = self.init_r[total_relation]
            pre_emb = F.normalize(history_embs[-1]) if self.layer_norm else history_embs[-1]
            total_entity_embs = pre_emb[total_entity]
            concatenated_embedding = torch.cat((total_entity_embs, total_relation_embs), dim=1)
            query_embedding = torch.mm(concatenated_embedding, self.mlp_weight) + self.mlp_bias
            query_embedding = torch.tanh(query_embedding)
            # 结束
        else:
            # 使用子图采样
            non_zero = [key for key in history_index.keys() if len(history_index[key]) != 0]
            entity_history_emb = []
            entity_history_dgl_tensor = None
            for history_idex in range(0, number):
                entity_history_emb.append(local_h.clone())
            for key in non_zero:
                for i in history_index[key]:
                    if entity_history_dgl_tensor is None:
                        entity_history_dgl_tensor = batched_glist[i].ndata['h'][key].unsqueeze(0)
                    else:
                        temp = batched_glist[i].ndata['h'][key].unsqueeze(0)

                        entity_history_dgl_tensor = torch.cat((entity_history_dgl_tensor, temp), 0)
                for buzu in range(0, number - len(history_index[key])):
                    entity_history_dgl_tensor = torch.cat(
                        (entity_history_dgl_tensor, entity_history_dgl_tensor[-1].unsqueeze(0)))

            if entity_history_dgl_tensor is not None:
                assert (len(non_zero) * number == entity_history_dgl_tensor.size()[0])

            used_cal = self.h.clone()

            entity_history_dgl_tensor = entity_history_dgl_tensor.view(-1, number, 200)
            entity_first_one = used_cal[non_zero].view(-1, 1, 200)
            cat_entity = torch.cat((entity_first_one, entity_history_dgl_tensor), dim=1)
            repeated_tensor_time = []
            for key in entity_history_index_filtered:
                entity_history_index_filtered[key].append(entity_history_index_filtered[key][0]-1)
                entity_history_index_filtered[key] = sorted(entity_history_index_filtered[key])
                if len(entity_history_index_filtered[key]) != number+1:
                    for j in range(number+1-len(entity_history_index_filtered[key])):
                        entity_history_index_filtered[key].append(entity_history_index_filtered[key][-1])
                entity_history_index_filtered[key] = sorted(entity_history_index_filtered[key])
                entity_history_index_filtered[key] = torch.from_numpy(np.array(entity_history_index_filtered[key])).cuda() if use_cuda else torch.from_numpy(np.array(entity_history_index_filtered[key]))
                time_emb = self.get_time_embedding(train_sample_num, entity_history_index_filtered[key])
                repeated_tensor_time.append(time_emb)

            repeated_tensor_time = torch.stack(repeated_tensor_time, dim=0)

            finally_tensor = cat_entity + repeated_tensor_time
            # 划分batch
            batch_size = 128
            total_samples = finally_tensor.shape[0]
            num_batches = (total_samples + batch_size - 1) // batch_size
            batched_tensors = []
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_samples)
                batched_tensor = finally_tensor[start_idx:end_idx, :, :]
                batched_tensors.append(batched_tensor)
            sample_total_tensors = []
            for i in range(len(batched_tensors)):
                output = self.transformer_encoder(batched_tensors[i])
                sample_total_tensors.append(output)

            output = torch.cat(sample_total_tensors, dim=0)
            # output = self.transformer_encoder(finally_tensor)
            #used_cal[non_zero] = output[:, 3, :]
            output = output[:, 1:number+1, :]
            output = output.reshape(output.size()[0]*number, 200)
            for idx, history_emb in enumerate(entity_history_emb):  # 这步是提升的关键，进行赋值
                history_emb[non_zero] = output[idx::number]
            for current_g in entity_history_emb:
                current_h = F.normalize(current_g) if self.layer_norm else current_g
                time_weight = F.sigmoid(torch.mm(current_h, self.time_gate_weight) + self.time_gate_bias)
                used_cal = time_weight * current_h + (1 - time_weight) * self.gruh
                history_embs.append(used_cal)
                calculate_embs.append(used_cal)
            #新结束

            # 不使用transformer
            # for idx, history_emb in enumerate(entity_history_emb):  # 这步是提升的关键，进行赋值
            #     history_emb[non_zero] = entity_history_dgl_tensor[idx::3]
            #
            # # current_h = torch.zeros((entity_history_emb[0].size()[0], entity_history_emb[0].size()[1])).cuda().to(self.gpu)
            # # for current_g in entity_history_emb:
            # #     current_h += F.normalize(current_g) if self.layer_norm else current_g
            # current_h = entity_history_emb[-1]
            # time_weight = F.sigmoid(torch.mm(current_h, self.time_gate_weight) + self.time_gate_bias)
            # used_cal = time_weight * current_h + (1 - time_weight) * self.gruh
            # history_embs.append(used_cal)
            # calculate_embs.append(used_cal)

            total_relation_embs = self.init_r[total_relation]
            pre_emb = F.normalize(history_embs[-1]) if self.layer_norm else history_embs[-1]
            total_entity_embs = pre_emb[total_entity]
            concatenated_embedding = torch.cat((total_entity_embs, total_relation_embs), dim=1)
            query_embedding = torch.mm(concatenated_embedding, self.mlp_weight) + self.mlp_bias
            query_embedding = torch.tanh(query_embedding)

            # return history_embs, static_emb, self.init_r, gate_list, degree_list
        return history_embs, static_emb, self.init_r, gate_list, degree_list, query_embedding


    def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda, history_index, zero_tensor_head, total_entity, total_relation, entity_history_index_filtered, train_sample_num, number, Dynic):
    # def predict(self, test_graph, num_rels, static_graph, test_triplets, use_cuda, history_index, entity_history_index_filtered, train_sample_num):

        with torch.no_grad():

            inverse_test_triplets = test_triplets[:, [2, 1, 0]]
            inverse_test_triplets[:, 1] = inverse_test_triplets[:, 1] + num_rels  # 将逆关系换成逆关系的id
            all_triples = torch.cat((test_triplets, inverse_test_triplets))

            evolve_embs, _, r_emb, _, _, query_embedding = self.forward(test_graph, static_graph, use_cuda, history_index, total_entity, total_relation, entity_history_index_filtered, train_sample_num, number, Dynic)
            # evolve_embs, _, r_emb, _, _ = self.forward(test_graph, static_graph, use_cuda, history_index, entity_history_index_filtered, train_sample_num)

            embedding = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

            # statics_query_embedding = torch.nn.functional.softmax(query_embedding, dim=1)
            # score_sta = statics_query_embedding + zero_tensor_head

            statics_query_embedding = query_embedding + zero_tensor_head
            score_sta = torch.nn.functional.softmax(statics_query_embedding, dim=1)

            # view_new = query_embedding.cpu().numpy()
            # 绘制嵌入向量的值
            # plt.plot(view_new, marker='s', linestyle='-')
            # plt.title('Visualization of Embedding Vector')
            # plt.xlabel('Index')
            # plt.ylabel('Value')
            # plt.show()

            score = self.decoder_ob.forward(embedding, r_emb, all_triples, mode="test")
            score = score + score_sta
            # # score[0, 1570] = 35
            # prediction_array = np.squeeze(score)
            # threshold = 0.5  # 根据你的需求设置阈值
            #
            # # 过滤得分小于阈值的值
            # filtered_indices = np.where(prediction_array.cpu() > threshold)[0]
            # filtered_values = prediction_array[filtered_indices].cpu()
            #
            # # 创建横坐标
            # x_values = filtered_indices + 1  # 索引从1开始
            #
            # # 绘制条形统计图，设置条形宽度为1
            # plt.bar(x_values, filtered_values, width=1)
            # # 设置横坐标和纵坐标的标签
            # plt.xlabel('Entity', fontsize=16)
            # plt.ylabel('Score', fontsize=16)
            # # 设置横坐标刻度的字体大小
            # plt.xticks(fontsize=12)
            # # 设置纵坐标刻度的字体大小
            # plt.yticks(fontsize=12)
            # # 显示图形
            # plt.show()

            score_rel = self.rdecoder.forward(embedding, r_emb, all_triples, mode="test")
            return all_triples, score, score_rel

    def get_relation_vec(self):
        relation_vec = {}
        for relation in range(self.num_rels * 2):
            relation_vec[relation] = self.emb_rel[relation, :].cpu().detach().numpy()
        relation_vec[self.num_rels*2] = np.zeros(200)
        with open('../data/ICEWS14/relation2vec_RGCN.json', 'w', encoding='utf-8') as f:
            json.dump(relation_vec, f, ensure_ascii=False, cls=NpEncoder)

    def get_loss(self, glist, triples, static_graph, use_cuda, history_index, zero_tensor_head, total_entity, total_relation, entity_history_index_filtered, train_sample_num, number, Dynic):
    # def get_loss(self, glist, triples, static_graph, use_cuda, history_index, entity_history_index_filtered, train_sample_num):

        """
        :param glist:
        :param triplets:
        :param static_graph:
        :param use_cuda:
        :return:
        """
        loss_ent = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_rel = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)
        loss_static = torch.zeros(1).cuda().to(self.gpu) if use_cuda else torch.zeros(1)

        inverse_triples = triples[:, [2, 1, 0]]
        inverse_triples[:, 1] = inverse_triples[:, 1] + self.num_rels
        all_triples = torch.cat([triples, inverse_triples])
        all_triples = all_triples.to(self.gpu)

        # evolve_embs, static_emb, r_emb, _, _ = self.forward(glist, static_graph, use_cuda, history_index, entity_history_index_filtered, train_sample_num)

        evolve_embs, static_emb, r_emb, _, _, query_embedding = self.forward(glist, static_graph, use_cuda, history_index, total_entity, total_relation, entity_history_index_filtered, train_sample_num, number, Dynic)

        pre_emb = F.normalize(evolve_embs[-1]) if self.layer_norm else evolve_embs[-1]

        statics_query_embedding = query_embedding + zero_tensor_head
        score_sta = torch.nn.functional.softmax(statics_query_embedding, dim=1)

        # statics_query_embedding = torch.nn.functional.softmax(query_embedding, dim=1)
        # score_sta = statics_query_embedding + zero_tensor_head

        if self.entity_prediction:
            scores_ob = self.decoder_ob.forward(pre_emb, r_emb, all_triples).view(-1, self.num_ents)
            scores_ob = scores_ob + score_sta
            loss_ent += self.loss_e(scores_ob, all_triples[:, 2])

        if self.relation_prediction:
            score_rel = self.rdecoder.forward(pre_emb, r_emb, all_triples, mode="train").view(-1, 2 * self.num_rels)
            loss_rel += self.loss_r(score_rel, all_triples[:, 1])

        if self.use_static:
            if self.discount == 1:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180) * (time_step + 1)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
            elif self.discount == 0:
                for time_step, evolve_emb in enumerate(evolve_embs):
                    step = (self.angle * math.pi / 180)
                    if self.layer_norm:
                        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
                    else:
                        sim_matrix = torch.sum(static_emb * evolve_emb, dim=1)
                        c = torch.norm(static_emb, p=2, dim=1) * torch.norm(evolve_emb, p=2, dim=1)
                        sim_matrix = sim_matrix / c
                    mask = (math.cos(step) - sim_matrix) > 0
                    loss_static += self.weight * torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_ent, loss_rel, loss_static
