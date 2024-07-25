import csv
import json
import math
import os
import pickle
import random
from collections import defaultdict
from itertools import permutations
import logging

import dgl
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import IterableDataset, DataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)




#  region 。。。
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, text_c=None, sen_emo=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        # self.text_b = text_b
        # self.text_c = text_c
        self.label = label
        self.sen_emo = sen_emo


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, speaker_ids, mention_ids, sentiment_ids,
                 emotion_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.speaker_ids = speaker_ids
        self.mention_ids = mention_ids
        self.sentiment_ids = sentiment_ids
        self.emotion_ids = emotion_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


# endregion

class bertsProcessor(DataProcessor):  # bert_s
    def __init__(self, src_file, n_class):
        random.seed(42)
        self.D = [[], [], []]  # 将整个数据集进行处理
        emotion_dict = {}
        sentiment_dict = {}

        with open(src_file + '/speakers.txt', "r", encoding="utf8") as file:
            speakers_list = [line.strip() for line in file.readlines()]

        for sid in range(3):
            with open(src_file + ["/train.json", "/dev.json", "/test.json"][sid], "r", encoding="utf8") as f:
                data = json.load(f)
            # if sid == 0:
            #     random.shuffle(data)
            data_type = ["train", "dev", "test"][sid]
            not_speaker = []
            for speaker in tqdm(speakers_list, desc=f"Speakers_{data_type}", unit="item"):
                if speaker in not_speaker:
                    continue
                one_speaker = []
                for i in range(len(data)):
                    for key in data[i][2]:
                        if key == speaker:
                            sen_emo_one_hot_list = []
                            sen_emo_list = []
                            sen_emo_cl = []
                            sen_promt , emo_promt = [], []
                            for k in range(len(data[i][1])):
                                emotion_one_hot = torch.nn.functional.one_hot(
                                    torch.tensor(emotion_dict[data[i][1][k]['Emotion']]), len(emotion_dict))
                                sentiment_one_hot = torch.nn.functional.one_hot(
                                    torch.tensor(sentiment_dict[data[i][1][k]['Sentiment']]), len(sentiment_dict))
                                sen_emo_one_hot = torch.cat([sentiment_one_hot, emotion_one_hot])
                                sen_emo_one_hot_tolist = sen_emo_one_hot.tolist()
                                sen_emo_one_hot_tolist = [x + 1 for x in sen_emo_one_hot_tolist]  # 以1开始
                                sen_emo_one_hot_list.append(sen_emo_one_hot_tolist)
                                sen, emo = data[i][1][k]['Sentiment'], data[i][1][k]['Emotion']

                                sen_emo = f'{sen}-*-{emo}'
                                sen_promt.append(sen)
                                emo_promt.append(emo)
                                sen_emo_cl.append(sen_emo)
                                sen_emo = []

                                sen_emo_list.append(sen_emo)

                            prc_id = []
                            for k in ["Neuroticism", "Extraversion", "Openness", "Agreeableness", "Conscientiousness"]:
                                if data[i][2][key][k] == 'high':
                                    prc_id += [1]
                                else:
                                    prc_id += [0]

                            d = data[i][0]
                            for j in range(len(data[i][2])):
                                d = [('[key]' if item == key else f'[unused{j + 1}]' if item == list(data[i][2].keys())
                                [j] else item)
                                     if idx % 2 == 0 else item
                                     for idx, item in
                                     enumerate(segment.strip() for item in d for segment in item.split(': '))]


                            d = [item if idx % 2 == 0 else ''.join(
                                str(n) for n in sen_promt[int((idx - 1) / 2)]) + '-' + item
                                 for idx, item in enumerate(d)]


                            d = [d, prc_id, sen_emo_cl]
                            one_speaker.append(d)

                if one_speaker:
                    self.D[sid] += [one_speaker]

        logger.info(str(len(self.D[0])) + "," + str(len(self.D[1])) + "," + str(len(self.D[2])))

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[0], "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[1], "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.D[2], "test")

    def get_labels(self):
        """See base class."""
        return [str(x) for x in range(2)]

    def _create_examples(self, data, set_type):
        examples_list = []
        """Creates examples for the training and dev sets."""
        for (idx, dd) in enumerate(data):
            examples = []
            # guid, label,text_a(对话)，text_b/text_c实体对
            for (i, d) in enumerate(dd):
                guid = "%s-%s-%s" % (set_type, idx, i)  # 唯一标识符
                text_a = d[0]
                label = d[1]
                sen_emo = d[2]
                # sen_emo_list = data[i][2]
                # examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=data[i][3], text_c=text_c))
                # examples.append(InputExample(guid=guid, text_a=text_a, text_b=sen_emo_list, label=label))
                examples.append(InputExample(guid=guid, text_a=text_a, label=label, sen_emo=sen_emo))
            examples_list.append(examples)
        return examples_list


def tokenize(text, sen_emo, tokenizer, start_mention_id):  # text = example.text_a
    speaker2id = {'[key]': 1, '[unused1]': 2, '[unused2]': 3, '[unused3]': 4, '[unused4]': 5, '[unused5]': 6}
    emotion_dict = {"anger": 0, "astonished": 1, "depress": 2, "disgust": 3, "fear": 4, "grateful": 5, "happy": 6,
                    "negative-other": 7, "neutral": 8, "positive-other": 9, "relaxed": 10, "sadness": 11,
                    "worried": 12}
    sentiment_dict = {"negative": 0, "neutral": 1, "positive": 2}
    textraw = text
    text = []  # 用来装token
    speaker_ids = []
    mention_ids = []
    sentiment_ids = []
    emotion_ids = []
    mention_id = start_mention_id  # 以用户为单位 区别实体与后续实体
    speaker_id = 0
    for t in textraw:
        if t in ['[key]', '[unused1]', '[unused2]', '[unused3]', '[unused4]', '[unused5]']:
            speaker_id = speaker2id[t]
            mention_id += 1

        else:

            tokens = tokenizer.tokenize(t)
            sentiment, emotion = sen_emo[mention_id - 1].split('-*-')
            for tok in tokens:
                text += [tok]
                speaker_ids.append(speaker_id)
                mention_ids.append(mention_id)
                sentiment_ids.append(sentiment_dict[sentiment])
                emotion_ids.append(emotion_dict[emotion])

    return text, speaker_ids, mention_ids, sentiment_ids, emotion_ids




def convert_examples_to_features(examples_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    print("#examples", len(examples_list))

    features_list = []
    if CPED == True:
        for examples in examples_list:
            features = [[]]
            for (ex_index, example) in enumerate(examples):
                tokens_a, tokens_a_speaker_ids, tokens_a_mention_ids, tokens_a_sentiment_ids, tokens_a_emotion_ids = tokenize(
                    example.text_a, example.sen_emo, tokenizer, 0)

                _truncate_seq_tuple(tokens_a=tokens_a, max_length=max_seq_length - 2,
                                    tokens_a_speaker_ids=tokens_a_speaker_ids,
                                    tokens_a_mention_ids=tokens_a_mention_ids,
                                    sentiment_ids=tokens_a_sentiment_ids,
                                    emotion_ids=tokens_a_emotion_ids)

                tokens = []
                segment_ids = []
                speaker_ids = []
                mention_ids = []
                sentiment_ids = []
                emotion_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                speaker_ids.append(0)
                mention_ids.append(0)
                sentiment_ids.append(0)
                emotion_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)
                speaker_ids = speaker_ids + tokens_a_speaker_ids
                mention_ids = mention_ids + tokens_a_mention_ids
                sentiment_ids = sentiment_ids + tokens_a_sentiment_ids
                emotion_ids = emotion_ids + tokens_a_emotion_ids
                tokens.append("[SEP]")
                segment_ids.append(0)
                speaker_ids.append(0)
                mention_ids.append(0)
                sentiment_ids.append(0)
                emotion_ids.append(0)



                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                tokens = [str(element) for element in tokens]

                input_mask = [1] * len(input_ids)

                while len(input_ids) < max_seq_length:
                    input_ids.append(0)
                    input_mask.append(0)
                    segment_ids.append(0)
                    speaker_ids.append(0)
                    mention_ids.append(0)
                    sentiment_ids.append(0)
                    emotion_ids.append(0)
                assert len(input_ids) == max_seq_length
                assert len(input_mask) == max_seq_length
                assert len(segment_ids) == max_seq_length
                assert len(speaker_ids) == max_seq_length
                assert len(mention_ids) == max_seq_length
                assert len(sentiment_ids) == max_seq_length
                assert len(emotion_ids) == max_seq_length

                label_id = example.label


                if 1 not in speaker_ids:
                    continue
                features[-1].append(
                    InputFeatures(
                        input_ids=input_ids,  # input_ids: cls dialog sep a1(实体) sep a2(实体) sep pad...
                        input_mask=input_mask,  # pad对应0 ，有内容的对应1
                        segment_ids=segment_ids,  # 实体对应1，其他对应0
                        label_id=label_id,  # label 的one-hot
                        speaker_ids=speaker_ids,  # 每个token对应的speaker
                        mention_ids=mention_ids,
                        sentiment_ids=sentiment_ids,
                        emotion_ids=emotion_ids))  # 每个实体对应的id
                if len(features[-1]) == 1:  # 判断最后一个元素是否填充内容
                    features.append([])

            if len(features[-1]) == 0:  # 去掉最后一个无内容的列表
                features = features[:-1]
            features_list.append(features)
        print('#features', len(features_list))
        return features_list


def _truncate_seq_tuple(tokens_a='', tokens_b='', tokens_c='', max_length='', tokens_a_speaker_ids='',
                        tokens_b_speaker_ids='', tokens_c_speaker_ids='', tokens_a_mention_ids='',
                        sentiment_ids='', emotion_ids=''):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()  # 删除 tokens_a的最后一个元素
            tokens_a_speaker_ids.pop()
            tokens_a_mention_ids.pop()
            sentiment_ids.pop()
            emotion_ids.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
            tokens_b_speaker_ids.pop()
        else:
            tokens_c.pop()
            tokens_c_speaker_ids.pop()


def get_cuda(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    return tensor


def mention2mask(mention_id):
    slen = len(mention_id)
    mask = []
    turn_mention_ids = [i for i in range(1, np.max(mention_id) - 1)]
    for j in range(len(mention_id)):
        tmp = None
        if mention_id[j] not in turn_mention_ids:
            tmp = np.zeros(len(mention_id), dtype=bool)
            tmp[j] = 1
        else:
            start = mention_id[j]
            end = mention_id[j]

            mention_id_tmp = mention_id[j]
            while mention_id_tmp - 1 in turn_mention_ids:
                mention_id_tmp = mention_id_tmp - 1
                start = mention_id_tmp

            tmp = (mention_id >= start) & (mention_id <= end)
        mask.append(tmp)
    mask = np.stack(mask)
    return mask


class TUCOREGCNDataset(IterableDataset):

    def __init__(self, src_file, save_file, max_seq_length, tokenizer, n_class, encoder_type):

        super(TUCOREGCNDataset, self).__init__()

        self.data = None
        self.input_max_length = max_seq_length

        print('Reading data from {}.'.format(src_file))
        if os.path.exists(save_file):
            with open(file=save_file, mode='rb') as fr:
                info = pickle.load(fr)
                self.data = info['data']
            print('load preprocessed data from {}.'.format(save_file))

        else:
            self.data = []

            bertsProcessor_class = bertsProcessor(src_file, n_class)
            if "train" in save_file:
                examples = bertsProcessor_class.get_train_examples(save_file)
            elif "dev" in save_file:
                examples = bertsProcessor_class.get_dev_examples(save_file)
            elif "test" in save_file:
                examples = bertsProcessor_class.get_test_examples(save_file)
            else:
                print('error')

            if encoder_type == "BERT":
                features_list = convert_examples_to_features(examples, max_seq_length, tokenizer)
            else:
                features = convert_examples_to_features_roberta(examples, max_seq_length, tokenizer)

            for features in features_list:
                f_list = []

                for f in features:
                    speaker_infor = self.make_speaker_infor(f[0].speaker_ids, f[0].mention_ids)
                    turn_node_num = max(f[0].mention_ids)
                    entity_edges_infor = self.make_entity_edges_infor(f[0].input_ids,
                                                                      f[0].mention_ids)

                    graph, used_mention = self.create_graph(speaker_infor, turn_node_num, entity_edges_infor)

                    assert len(used_mention) == (max(f[0].mention_ids) + 1 + 1)

                    f_list.append({
                        'input_ids': np.array(f[0].input_ids),
                        'segment_ids': np.array(f[0].segment_ids),
                        'input_mask': np.array(f[0].input_mask),
                        'speaker_ids': np.array(f[0].speaker_ids),
                        'label_ids': np.array(f[0].label_id),
                        'mention_id': np.array(f[0].mention_ids),
                        'turn_mask': mention2mask(np.array(f[0].mention_ids)),
                        'graph': graph,
                        'sentiment_ids': np.array(f[0].sentiment_ids),
                        'emotion_ids': np.array(f[0].emotion_ids)
                    })
                self.data.append(f_list)
            # save data
            with open(file=save_file, mode='wb') as fw:
                pickle.dump({'data': self.data}, fw)
            print('finish reading {} and save preprocessed data to {}.'.format(src_file, save_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __iter__(self):
        return iter(self.data)

    def turn2speaker(self, turn):
        return turn.split()[1]

    def make_speaker_infor(self, speaker_id, mention_id):
        tmp = defaultdict(set)
        for i in range(1, len(speaker_id)):
            if speaker_id[i] == 0:
                break
            tmp[speaker_id[i]].add(mention_id[i])

        speaker_infor = dict()
        for k, va in tmp.items():
            speaker_infor[k] = list(va)
        return speaker_infor

    def make_entity_edges_infor(self, input_ids, mention_id):
        entity_edges_infor = {'h': [], 't': []}
        head_mention_id = max(mention_id) - 1
        tail_mention_id = max(mention_id)
        head = list()
        tail = list()
        for i in range(len(mention_id)):
            if mention_id[i] == head_mention_id:
                head.append(input_ids[i])

        for i in range(len(mention_id)):
            if mention_id[i] == tail_mention_id:
                tail.append(input_ids[i])

        for i in range(len(input_ids) - len(head)):

            if input_ids[i:i + len(head)] == head:
                entity_edges_infor['h'].append(mention_id[i])

        for i in range(len(input_ids) - len(tail)):
            if input_ids[i:i + len(tail)] == tail:
                entity_edges_infor['t'].append(mention_id[i])

        return entity_edges_infor

    def create_graph(self, speaker_infor, turn_node_num, entity_edges_infor, head_mention_id=None,
                     tail_mention_id=None):
        d = defaultdict(list)
        used_mention = set()
        for i, mentions in speaker_infor.items():
            for h, t in permutations(mentions, 2):
                d[('node', 'speaker', 'node')].append((h, t))
                used_mention.add(h)
                used_mention.add(t)
            if i == 1:
                speaker_node = turn_node_num + 1
                for j in mentions:
                    d[('node', 'main_speaker', 'node')].append((speaker_node, j))
                    d[('node', 'main_speaker', 'node')].append((j, speaker_node))
                    used_mention.add(speaker_node)
                    used_mention.add(j)

        if d[('node', 'speaker', 'node')] == []:
            d[('node', 'speaker', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)

        # add dialog edges
        for i in range(1, turn_node_num + 1):
            d[('node', 'dialog', 'node')].append((i, 0))
            d[('node', 'dialog', 'node')].append((0, i))
            used_mention.add(i)
            used_mention.add(0)
        if d[('node', 'dialog', 'node')] == []:
            d[('node', 'dialog', 'node')].append((1, 0))
            used_mention.add(1)
            used_mention.add(0)
        assert len(used_mention) == (turn_node_num + 1 + 1)

        graph = dgl.heterograph(d)

        return graph, used_mention


class TUCOREGCNDataloader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle=False, relation_num=36, max_length=512, drop_last=False):
        super(TUCOREGCNDataloader, self).__init__(dataset, batch_size=batch_size, drop_last=drop_last)
        self.shuffle = shuffle
        self.length = len(self.dataset)
        self.max_length = max_length

        self.relation_num = relation_num

        self.order = list(range(self.length))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.order)
            self.data = [self.dataset[idx] for idx in self.order]
        else:
            self.data = self.dataset
        batch_num = math.ceil(self.length / self.batch_size)
        self.batches = [self.data[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                        for idx in range(0, batch_num)]
        self.batches_order = [self.order[idx * self.batch_size: min(self.length, (idx + 1) * self.batch_size)]
                              for idx in range(0, batch_num)]

        for idx, minibatch in enumerate(self.batches):
            cur_bsz = len(minibatch[0])
            batch_size = cur_bsz
            # begin
            input_ids = torch.LongTensor(batch_size, self.max_length).cpu()
            segment_ids = torch.LongTensor(batch_size, self.max_length).cpu()
            input_mask = torch.LongTensor(batch_size, self.max_length).cpu()
            mention_id = torch.LongTensor(batch_size, self.max_length).cpu()
            speaker_id = torch.LongTensor(batch_size, self.max_length).cpu()
            turn_masks = torch.LongTensor(batch_size, self.max_length, self.max_length).cpu()
            label_ids = torch.Tensor(batch_size, self.relation_num).cpu()
            sentiment_ids = torch.Tensor(batch_size, self.max_length).cpu()
            emotion_ids = torch.Tensor(batch_size, self.max_length).cpu()



            for mapping in [input_ids, segment_ids, input_mask, mention_id, label_ids, turn_masks, speaker_id,
                            sentiment_ids, emotion_ids]:
                if mapping is not None:
                    mapping.zero_()

            graph_list = []


            for i, example in enumerate(minibatch[0]):
                mini_input_ids, mini_segment_ids, mini_input_mask, mini_label_ids, mini_mention_id, mini_speaker_id, turn_mask, graph, mini_sentiment_ids, mini_emotion_ids = \
                    example['input_ids'], example['segment_ids'], example['input_mask'], example['label_ids'], \
                        example['mention_id'], example['speaker_ids'], example['turn_mask'], example['graph'], \
                        example['sentiment_ids'], example['emotion_ids']
                graph_list.append(graph.to(torch.device('cuda:0')))

                word_num = mini_input_ids.shape[0]
                relation_num = mini_label_ids.shape[0]

                input_ids[i, :word_num].copy_(torch.from_numpy(mini_input_ids))
                segment_ids[i, :word_num].copy_(torch.from_numpy(mini_segment_ids))
                input_mask[i, :word_num].copy_(torch.from_numpy(mini_input_mask))
                mention_id[i, :word_num].copy_(torch.from_numpy(mini_mention_id))
                speaker_id[i, :word_num].copy_(torch.from_numpy(mini_speaker_id))
                turn_masks[i, :word_num, :word_num].copy_(torch.from_numpy(turn_mask))
                label_ids[i, :relation_num].copy_(torch.from_numpy(mini_label_ids))
                sentiment_ids[i, :word_num].copy_(torch.from_numpy(mini_sentiment_ids))
                emotion_ids[i, :word_num].copy_(torch.from_numpy(mini_emotion_ids))

            context_word_mask = input_ids > 0
            context_word_length = context_word_mask.sum(1)


            batch_max_length = int(context_word_length.max(dim=0)[0])

            # yield 逐个生成数据批次，而不是一次性将整个数据集加载到内存中，避免因内存不足而导致程序崩溃或性能下降
            yield {'input_ids': get_cuda(input_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'segment_ids': get_cuda(segment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'input_masks': get_cuda(input_mask[:cur_bsz, :batch_max_length].contiguous()),
                   'mention_ids': get_cuda(mention_id[:cur_bsz, :batch_max_length].contiguous()),
                   'speaker_ids': get_cuda(speaker_id[:cur_bsz, :batch_max_length].contiguous()),
                   'label_ids': get_cuda(label_ids[:cur_bsz, :self.relation_num].contiguous()),
                   'turn_masks': get_cuda(turn_masks[:cur_bsz, :batch_max_length, :batch_max_length].contiguous()),
                   'graphs': graph_list,
                   'sentiment_ids': get_cuda(sentiment_ids[:cur_bsz, :batch_max_length].contiguous()),
                   'emotion_ids': get_cuda(emotion_ids[:cur_bsz, :batch_max_length].contiguous())
                   }
