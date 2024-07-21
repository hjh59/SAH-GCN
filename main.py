# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import argparse
import random

from tqdm import tqdm
import numpy as np
import torch

from models.BERT import tokenization
from models.BERT.TUCOREGCN_BERT import BertConfig

from models.BERT.SAH_GCN import SAH_GCN

from models.optimization import BertAdam

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score

from data_SAH_GCN import TUCOREGCNDataset, TUCOREGCNDataloader


def calc_test_result(logits_all, labels_all, threshold=None, type=type):
    # 模型预测指标
    threshold = []

    val_predicts = (np.array(logits_all) >= threshold).astype(int)
    val_targets = np.array(labels_all).astype(int)

    f1_score_micro = f1_score(val_targets, val_predicts, average='micro')
    f1_score_macro = f1_score(val_targets, val_predicts, average='macro')
    acc_single_list = acc_single(threshold, logits_all, labels_all, type=type)
    ave_acc_score = sum(acc_single_list) / len(acc_single_list)

    target_names = {'Neu.': 0, 'Ext.': 1, 'Ope.': 2, 'Agr.': 3, 'Con.': 4}
    print(
        classification_report(val_targets, val_predicts, zero_division='warn', digits=4, target_names=target_names))

    print(f"Single accuracy Score  = {acc_single_list}")
    print(f"Average accuracy Score = {ave_acc_score}")
    print(f"F1 Score (Macro) = {f1_score_macro}")

    eval_f1 = f1_score_macro
    eval_acc = ave_acc_score
    return eval_f1, eval_acc, threshold



def model_eval(model, data_loader, desc, threshold):
    # model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
    model.eval()
    loss_total = 0
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    logits_all = []
    labels_all = []
    # predict_all = np.array([], dtype=int)
    # labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for step, batch in enumerate(tqdm(data_loader, desc=desc)):
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            input_masks = batch['input_masks'].to(device)
            mention_ids = batch['mention_ids'].to(device)
            speaker_ids = batch['speaker_ids'].to(device)
            label_ids = batch['label_ids'].to(device)
            turn_mask = batch['turn_masks'].to(device)
            graphs = batch['graphs']
            sentiment_ids = batch['sentiment_ids'].to(device)
            emotion_ids = batch['emotion_ids'].to(device)

            is_single = False
            label_ids = label_ids[0:1, :]
            label_ids = label_ids

            tmp_eval_loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks,
                                          speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids,
                                          labels=label_ids, turn_mask=turn_mask, max_seq_length=max_seq_length,
                                          is_single=is_single, sentiment_ids=sentiment_ids, emotion_ids=emotion_ids)
            model.zero_grad()
            # label_ids = label_ids[0:1, :]
            loss_total += tmp_eval_loss
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.cpu().numpy()
            logits_all.extend(logits.tolist())
            labels_all.extend(label_ids.tolist())
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
        eval_f1, eval_acc, threshold = calc_test_result(logits_all, labels_all, threshold=threshold, type=type)
    return eval_f1, eval_acc, threshold


def main():
    if os.path.exists(output_dir) and 'model.pt' in os.listdir(output_dir):
        if do_train:
            raise ValueError("Output directory ({}) already exists and is not empty.".format(output_dir))
    else:
        os.makedirs(output_dir, exist_ok=True)

    model = SAH_GCN(config=config, num_labels=5, max_length=max_seq_length).to(device)
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)

    if do_train:
        train_set = TUCOREGCNDataset(src_file=data_dir,
                                     save_file=data_dir + "/train_" + encoder_type + ".pkl",
                                     max_seq_length=max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                     encoder_type='BERT')
        train_loader = TUCOREGCNDataloader(dataset=train_set, batch_size=train_batch_size, shuffle=False,
                                           relation_num=n_class, max_length=max_seq_length, drop_last=False)

        dev_set = TUCOREGCNDataset(src_file=data_dir,
                                   save_file=data_dir + "/dev_" + encoder_type + ".pkl",
                                   max_seq_length=max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                   encoder_type=encoder_type)
        dev_loader = TUCOREGCNDataloader(dataset=dev_set, batch_size=eval_batch_size, shuffle=False,
                                         relation_num=n_class, max_length=max_seq_length)
    if do_test:
        test_set = TUCOREGCNDataset(src_file=data_dir,
                                    save_file=data_dir + "/test_" + encoder_type + ".pkl",
                                    max_seq_length=max_seq_length, tokenizer=tokenizer, n_class=n_class,
                                    encoder_type=encoder_type)
        test_loader = TUCOREGCNDataloader(dataset=test_set, batch_size=eval_batch_size, shuffle=False,
                                          relation_num=n_class, max_length=max_seq_length)

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_metric, best_threshold = 0, None
    if args.do_train:
        for epoch in range(num_epochs):
            model.train()
            print('Epoch [{}/{}]'.format(epoch + 1, num_epochs))
            for step, batch in enumerate(tqdm(train_loader, desc="Training")):
                input_ids = batch['input_ids'].to(device)
                segment_ids = batch['segment_ids'].to(device)
                input_masks = batch['input_masks'].to(device)
                mention_ids = batch['mention_ids'].to(device)
                speaker_ids = batch['speaker_ids'].to(device)
                label_ids = batch['label_ids'].to(device)
                turn_mask = batch['turn_masks'].to(device)
                graphs = batch['graphs']
                sentiment_ids = batch['sentiment_ids'].to(device)
                emotion_ids = batch['emotion_ids'].to(device)

                is_single = False
                label_ids = label_ids[0:1, :]
                label_ids = label_ids

                loss, logits = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_masks,
                                     speaker_ids=speaker_ids, graphs=graphs, mention_ids=mention_ids,
                                     labels=label_ids, turn_mask=turn_mask, max_seq_length=max_seq_length,
                                     is_single=is_single, sentiment_ids=sentiment_ids, emotion_ids=emotion_ids)
                model.zero_grad()
                loss.backward()
                optimizer.step()


            if args.do_eval:
                eval_f1, eval_acc, threshold = model_eval(model, dev_loader, "Dev", None)
            if args.eval_type == 'f1':
                if eval_acc >= best_metric:
                    torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                    best_metric = eval_f1
                    best_threshold = threshold
            if args.eval_type == 'acc':
                if eval_acc >= best_metric:
                    torch.save(model.state_dict(), os.path.join(output_dir, "model_best.pt"))
                    best_metric = eval_acc
                    best_threshold = threshold
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
        torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    if args.do_test:
        model.load_state_dict(torch.load(os.path.join(output_dir, "model_best.pt")))
        model_eval(model, test_loader, "Test", best_threshold)


def init_args():
    """定义参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu',
                        type=int,
                        default=0,
                        help="The number of the using gpu.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--data_dir",
                        default='datasets/CPED',
                        type=str,
                        required=False,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--config_file",
                        default='/home/hehaijun/47SAH_GCN/pre-trained_model/BERT/bert_config.json',
                        type=str,
                        required=False,
                        help="The config json file corresponding to the pre-trained model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--data_name",
                        default=None,
                        type=str,
                        required=False,
                        help="The name of the dataset to train.")
    parser.add_argument("--encoder_type",
                        default=None,
                        type=str,
                        required=False,
                        help="The type of pre-trained model.")
    parser.add_argument("--vocab_file",
                        default='/home/hehaijun/47SAH_GCN/pre-trained_model/BERT/vocab.txt',
                        type=str,
                        required=False,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--output_dir",
                        default='SAH-GCN_CPED_gpu_',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--learning_rate",
                        default=3e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--n_class",
                        default=5,
                        type=int,
                        help="The amount of the labels.")
    parser.add_argument('--seed',
                        type=int,
                        default=666,
                        help="random seed for initialization")
    parser.add_argument('--eval_type',
                        type=str,
                        default='acc',
                        help="The type to evaluate the model.")
    parser.add_argument("--single",
                        default=False,
                        action='store_true',
                        help="Whether to use f1 for dev evaluation during training.")
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = init_args()

    # region global variable
    cuda_number = args.gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_number)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # n_class = args.n_class
    n_class = 5
    data_dir = args.data_dir
    vocab_file = args.vocab_file
    config_file = args.config_file
    output_dir = args.output_dir + f'{args.gpu}'
    config = BertConfig.from_json_file(config_file)
    max_seq_length = args.max_seq_length
    encoder_type = 'BERT'
    num_epochs = args.num_train_epochs
    learning_rate = args.learning_rate
    do_train = args.do_train
    do_test = args.do_test
    set_seed(args.seed)

    type = 'multi'
    model_type = 'multi'
    torch.cuda.empty_cache()

    main()

    #  endregion
