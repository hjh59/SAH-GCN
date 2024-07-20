from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import torch.nn.functional as F

import dgl
import dgl.nn.pytorch as dglnn

from transformers import BertModel, BertConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    rel_names : list[str]
        Relation names.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 rel_names,
                 num_bases,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rel_names = rel_names
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
            for rel in rel_names
        })

        self.use_weight = weight
        self.use_basis = num_bases < len(self.rel_names) and weight
        if self.use_weight:
            if self.use_basis:
                self.basis = dglnn.WeightBasis((in_feat, out_feat), num_bases, len(self.rel_names))
            else:
                self.weight = nn.Parameter(torch.Tensor(len(self.rel_names), in_feat, out_feat))
                nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # bias
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def forward(self, g, inputs):
        """Forward computation
        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.
        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            weight = self.basis() if self.use_basis else self.weight
            wdict = {self.rel_names[i]: {'weight': w.squeeze(0)}
                     for i, w in enumerate(torch.split(weight, 1, dim=0))}
        else:
            wdict = {}
        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + torch.matmul(inputs[ntype], self.loop_weight)
            if self.bias:
                h = h + self.h_bias
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class TurnLevelLSTM(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_layers,
                 lstm_dropout,
                 dropout_rate):
        super(TurnLevelLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers,
                            dropout=lstm_dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.bilstm2hiddnesize = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, inputs):
        lstm_out = self.lstm(inputs)
        lstm_out = lstm_out[0].squeeze(0)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.bilstm2hiddnesize(lstm_out)
        return lstm_out


class SupConLoss(nn.Module):

    def __init__(self, temperature=0.5, scale_by_temperature=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.scale_by_temperature = scale_by_temperature

    def forward(self, features, labels=None, mask=None):
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels)
        features = F.normalize(features, p=2, dim=1)
        batch_size = features.shape[0]
        if labels is not None and mask is not None:  
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None: 
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:  
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
        '''
        示例: 
        labels: 
            tensor([[1.],
                    [2.],
                    [1.],
                    [1.]])
        mask:  # 两个样本i,j的label相等时，mask_{i,j}=1
            tensor([[1., 0., 1., 1.],
                    [0., 1., 0., 0.],
                    [1., 0., 1., 1.],
                    [1., 0., 1., 1.]]) 
        '''
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature) 
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = torch.exp(logits)

        logits_mask = torch.ones_like(mask).to(device) - torch.eye(batch_size).to(device)
        positives_mask = mask * logits_mask
        negatives_mask = 1. - mask

        num_positives_per_row = torch.sum(positives_mask, axis=1)  
        denominator = (torch.sum(exp_logits * negatives_mask, axis=1, keepdims=True) +
                       torch.sum(exp_logits * positives_mask, axis=1, keepdims=True))

        log_probs = logits - torch.log(denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")

        log_probs = (torch.sum(log_probs * positives_mask, axis=1)[num_positives_per_row > 0] /
                     num_positives_per_row[num_positives_per_row > 0])

        loss = -log_probs
        if self.scale_by_temperature:
            loss *= self.temperature
        loss = loss.mean()
        return loss


class SAH_GCN(nn.Module):
    def __init__(self, config, num_labels, gcn_layers, max_length, gcn_dropout):
        super(SAH_GCN, self).__init__()
        config = BertConfig()
        self.bert = BertModel(config)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.gcn_dim = config.hidden_size
        self.dim = config.hidden_size
        self.gcn_layers = gcn_layers
        self.num_labels = num_labels
        self.max_length = max_length
        self.activation = None

        self.classifier = nn.Linear(config.hidden_size * 1 * (self.gcn_layers), self.num_labels)
        self.classifier_CE = nn.Linear(config.hidden_size * 1 * (self.gcn_layers), self.num_labels * 2)
        self.classifier_speaker = nn.Linear(config.hidden_size * self.max_length, config.hidden_size)
        self.classifier_not_single = nn.Linear(config.hidden_size * 3 * (self.gcn_layers), config.hidden_size)

        self.CL = SupConLoss(temperature, scale_by_temperature=True)

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.turnAttention = nn.MultiheadAttention(embed_dim=config.hidden_size,
                                                   num_heads=config.num_attention_heads,
                                                   batch_first=True)

        self.rel_name_lists = ['speaker', 'dialog', 'main_speaker']

        self.GCN_layers = nn.ModuleList([RelGraphConvLayer(self.gcn_dim, self.gcn_dim, self.rel_name_lists,
                                                           num_bases=len(self.rel_name_lists),
                                                           activation=self.activation,
                                                           self_loop=True, dropout=gcn_dropout)
                                         for i in range(self.gcn_layers)])

        self.LSTM_layers = nn.ModuleList(
            [TurnLevelLSTM(hidden_size=config.hidden_size,
                           num_layers=,
                           lstm_dropout=,
                           dropout_rate=) for i in range(self.gcn_layers)])

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def get_speaker_output(self, speaker_ids, sequence_outputs, b_size, length):
        speaker_mask = (speaker_ids == 1).float()
        speaker_mask = speaker_mask.unsqueeze(-1)
        masked_sequence_output = sequence_outputs * speaker_mask
        if length != self.max_length:
            target_size = (b_size, self.max_length, self.dim)
            pad_sizes = []
            for i in range(len(target_size)):
                pad_before = max((target_size[i] - masked_sequence_output.size(i)) // 2, 0)
                pad_after = max(target_size[i] - masked_sequence_output.size(i) - pad_before, 0)
                pad_sizes.extend([0, pad_before + pad_after])
            masked_sequence_output = torch.nn.functional.pad(masked_sequence_output, pad_sizes)
        masked_sequence_output = masked_sequence_output.view(b_size, -1)
        speaker_outputs = self.classifier_speaker(masked_sequence_output)
        return speaker_outputs

    def get_affection_label(self, affection_id, mention_id):
        affection_list_fc = []
        init_sit = mention_id[0]
        for m in range(len(mention_id)):
            if mention_id[m] != init_sit:
                affection_list_fc.append(affection_id[m])
                init_sit = mention_id[m]
        affection_list_fc.pop()
        affection_list_fc = [int(aff) for aff in affection_list_fc]
        return affection_list_fc

    def overall_similarity(self, sentence_tensors):
        overall_sentence_vector = torch.mean(sentence_tensors, dim=0)
        similarities = F.cosine_similarity(overall_sentence_vector.unsqueeze(0), sentence_tensors, dim=1)
        overall_similarity_score = torch.mean(similarities).item()
        return overall_similarity_score

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, speaker_ids=None, graphs=None,
                mention_ids=None, labels=None, turn_mask=None, max_seq_length=512, model_type='train',
                pooled_output=None, is_single=False, sentiment_ids=None, emotion_ids=None, affect_ablation=False):

        if model_type == 'train':
            temp = {}
            similarity_list = []
            b_size = input_ids.size(0)
            length = input_ids.size(1)
            bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
            sequence_outputs = bert_output.last_hidden_state
            pooled_outputs = bert_output.pooler_output

            speaker_outputs = self.get_speaker_output(speaker_ids, sequence_outputs, input_ids.size(0),
                                                      input_ids.size(1))

            temp['bert1'], temp['bert2'], temp['bert3'] = sequence_outputs, pooled_outputs, speaker_outputs

            features, features_cl = None, None
            sentiment_labels, emotion_labels = None, None
            loss_cl_sen, loss_cl_emo = 0, 0
            turn_mask = turn_mask.repeat(12, 1, 1)
            turn_mask = torch.Tensor.bool(turn_mask)
            sequence_outputs, _ = self.turnAttention(query=sequence_outputs,
                                                     key=sequence_outputs,
                                                     value=sequence_outputs,
                                                     attn_mask=turn_mask)
            temp['turnattention1'], temp['turnattention2'] = sequence_outputs, _
            num_batch_turn = []
            slen = length

            for i in range(len(graphs)):
                sequence_output = sequence_outputs[i]
                mention_num = torch.max(mention_ids[i])
                num_batch_turn.append(mention_num + 1)
                # region get affection_label
                sentiment_list = self.get_affection_label(affection_id=sentiment_ids[i].tolist(),
                                                          mention_id=mention_ids[i].tolist())
                emotion_list = self.get_affection_label(affection_id=emotion_ids[i].tolist(),
                                                        mention_id=mention_ids[i].tolist())
                # endregion
                mention_index = get_cuda((torch.arange(mention_num) + 1).unsqueeze(1).expand(-1, slen))
                mentions = mention_ids[i].unsqueeze(0).expand(mention_num, -1)
                select_metrix = (mention_index == mentions).float()
                word_total_numbers = torch.sum(select_metrix, dim=-1).unsqueeze(-1).expand(-1, slen)
                select_metrix = torch.where(word_total_numbers > 0, select_metrix / word_total_numbers, select_metrix)
                x = torch.mm(select_metrix, sequence_output)
                x_cl = x
                x = torch.cat((pooled_outputs[i].unsqueeze(0), x), dim=0)
                x = torch.cat((x, speaker_outputs[i].unsqueeze(0)), dim=0)
                loss_cl_sen_single = self.CL(features=x_cl, labels=sentiment_list)
                if affect_ablation:
                    similarity_list.append(self.overall_similarity(x))

                if features is None:
                    features = x
                    features_cl = x_cl
                else:
                    features = torch.cat((features, x), dim=0)
                    features_cl = torch.cat((features_cl, x_cl), dim=0)
                if sentiment_labels is None:
                    sentiment_labels = sentiment_list
                    emotion_labels = emotion_list
                    loss_cl_sen = loss_cl_sen_single

                else:
                    sentiment_labels += sentiment_list
                    emotion_labels += emotion_list
                    loss_cl_sen += loss_cl_sen_single

            graph_big = dgl.batch(graphs)

            output_features = []

            for layer_num, GCN_layer in enumerate(self.GCN_layers):
                start = 0
                new_features = []
                for idx in num_batch_turn:
                    new_features.append(features[start])
                    lstm_out2 = self.LSTM_layers[layer_num](features[start + 1:start + idx].unsqueeze(0))
                    lstm_out = lstm_out2
                    new_features += lstm_out
                    new_features.append(features[start + idx])
                    start += idx
                features = torch.stack(new_features)
                features = GCN_layer(graph_big, {"node": features})["node"]
                output_features.append(features)

            graphs = dgl.unbatch(graph_big)

            graph_output = list()

            temp['graph'] = graphs
            fea_idx = 0
            for i in range(len(graphs)):
                node_num = graphs[i].number_of_nodes('node')
                intergrated_output = None
                speaker_node = fea_idx + node_num - 1
                for j in range(self.gcn_layers):
                    if intergrated_output == None:
                        intergrated_output = output_features[j][speaker_node]
                    else:
                        intergrated_output = torch.cat((intergrated_output, output_features[j][speaker_node]),
                                                       dim=-1)
                fea_idx += node_num
                graph_output.append(intergrated_output)
            graph_output = torch.stack(graph_output)

            temp['graph_output'] = graph_output

            speaker_outputs = graph_output
            pooled_outputs = speaker_outputs
            pooled_outputs = self.dropout(pooled_outputs)

            logits = self.classifier(pooled_outputs)
            if not is_single:
                logits = torch.min(logits, dim=0)[0]
            logits = logits.view(-1, self.num_labels)

            loss_fct = BCEWithLogitsLoss()
            labels = labels.view(-1, self.num_labels)
            loss = loss_fct(logits, labels)
            logits = torch.sigmoid(logits)
            loss += loss_cl_sen

            if affect_ablation is True:
                return loss, logits, similarity_list
            else:
                return loss, logits
