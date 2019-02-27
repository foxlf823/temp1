import torch
import torch.nn as nn  ## neural net library
import torch.optim as optim  # optimization package

import utils
from data_helpers import loadAbbreviations,load_dict,preprocessMentions,parserNcbiTxtFile_simple
from options import opt
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
import time
import logging
from pytorch_pretrained_bert import BertTokenizer, BertAdam, BertModel
from pytorch_pretrained_bert.modeling import BertPreTrainedModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info(opt)

if opt.random_seed != 0:
    random.seed(opt.random_seed)
    np.random.seed(opt.random_seed)
    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)

def parser_dict(dict):
    label_to_ix = {}

    labels = set([label for label in dict.id_to_names.keys()])
    for label in labels:
        label_to_ix[label] = len(label_to_ix) + 1  # 0 is for unknown
    label_to_ix['-1'] = 0
    return labels, label_to_ix

class NormDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        assert len(self.X) == len(self.Y), 'X and Y have different lengths'

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return(self.X[idx], self.Y[idx])

def getDictInstance(dict, label_to_ix, tokenizer):
    X = []
    Y = []
    for id, names in dict.id_to_names.items():
        for name in names:

            name_words = []

            instance = {}
            for word in tokenizer.tokenize(name):
                if word !='':
                    name_words.append(word)

            if len(name_words) == 0:
                continue

            name_words.insert(0, '[CLS]')
            name_words.append('[SEP]')
            instance['entity'] = tokenizer.convert_tokens_to_ids(name_words)
            # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
            segments_ids = [0]*len(instance['entity'])
            instance['sentence'] = segments_ids

            instance['feature'] = []

            X.append(instance)
            if id in label_to_ix.keys():
                Y.append(label_to_ix[id])

    set = NormDataset(X, Y)
    return set


def getNormInstance(documents, label_to_ix, tokenizer):
    X = []
    Y = []
    for doc in documents:
        for entity in doc.entities:
            entity_words = []
            entity_features =[]

            instance = {'entity': entity_words, 'feature':entity_features}
            for entity_word in tokenizer.tokenize(entity.text):
                if entity_word != '':
                    entity_words.append(entity_word)


            if len(entity_words) == 0:
                continue

            entity_words.insert(0, '[CLS]')
            entity_words.append('[SEP]')
            instance['entity'] = tokenizer.convert_tokens_to_ids(entity_words)

            #feature for output
            entity_features.append(entity.doc_id)
            entity_features.append(str(entity.start))
            entity_features.append(str(entity.end))
            entity_features.append(entity.text)
            entity_features.append("Disease")
            entity_features.append("preID")
            instance['feature'] = entity_features

            instance['sentence'] = [0]*len(instance['entity'])



            X.append(instance)

            if entity.gold_meshId in label_to_ix.keys():
                Y.append(label_to_ix[entity.gold_meshId])
            else:
                print('not id=', entity.gold_meshId)
                ab = label_to_ix['-1']
                Y.append(label_to_ix['-1'])


    set = NormDataset(X, Y)
    return set

def pad_sequence(x, max_len, eos_idx):
    padded_x = np.zeros((len(x), max_len), dtype=np.int)
    padded_x.fill(eos_idx)
    for i, row in enumerate(x):
        # assert eos_idx not in row, 'EOS in sequence {row}'
        padded_x[i][:len(row)] = row
    padded_x = torch.LongTensor(padded_x)

    return padded_x

def my_collate(batch):
    x, y = zip(*batch)
    # extract input indices
    x1 = [s['entity'] for s in x]
    x2 = [s['feature'] for s in x]
    x3 = [s['sentence'] for s in x]

    x1, x2, x3, mask, y = pad(x1, x2, x3, y, 0)

    if opt.gpu >= 0 and torch.cuda.is_available():
        x1 = x1.cuda(opt.gpu)
        x3 = x3.cuda(opt.gpu)
        mask = mask.cuda(opt.gpu)
        y = y.cuda(opt.gpu)
    return x1, x2, x3, mask, y

def pad(x1, x2, x3, y, eos_idx):

    entity_lengths = [len(row) for row in x1]
    max_entity_len = max(entity_lengths)

    # mask
    mask = [ [1]*length for length in entity_lengths]
    mask = pad_sequence(mask, max_entity_len, eos_idx)

    # entity
    padded_x = pad_sequence(x1, max_entity_len, eos_idx)
    # entity_lengths = torch.LongTensor(entity_lengths)

    #sentence
    padded_x3 = pad_sequence(x3, max_entity_len, eos_idx)

    y = torch.LongTensor(y).view(-1)


    return padded_x, x2, padded_x3, mask, y

def endless_get_next_batch(loaders, iters):
    try:
        inputs, features, sentences, mask, targets = next(iters)
    except StopIteration:
        iters = iter(loaders)
        inputs, features, sentences, mask, targets = next(iters)

    return inputs, features, sentences, mask, targets

class BertForSequenceClassification(BertPreTrainedModel):


    def __init__(self, config, num_labels):
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def evaluate(data_loader, model, label_to_ix):

    ix_to_label = {v:k for k,v in label_to_ix.items()}
    correct = 0
    total = 0
    model.eval()
    loader_it = iter(data_loader)
    num_it = len(data_loader)
    instances = []
    for j in range(num_it):
        mention_inputs, features, sentences,mask, targets = utils.endless_get_next_batch(data_loader, loader_it)
        pred = model(mention_inputs, sentences, mask)
        _, y_pred = torch.max(pred, 1)
        total += targets.size(0)
        # correct += (y_pred == targets).sum().sample_data[0]
        correct += (y_pred == targets).sum().item()
        # output evaluate
        pred_numpy = (y_pred.data).cpu().numpy()
        y_pred_labels = [ix_to_label[ix] for ix in pred_numpy]
        assert len(y_pred_labels)==len(features), 'y_pred_labels and features have different lengths'
        for i, pred_label in enumerate(y_pred_labels):
            features[i][5] = pred_label
            instances.append(features[i])

    acc = 100.0 * correct / total

    return acc, instances

if __name__ == "__main__":

    # load raw data
    traindocuments = parserNcbiTxtFile_simple(opt.train_file)
    devdocuments = parserNcbiTxtFile_simple(opt.dev_file)
    testdocuments = parserNcbiTxtFile_simple(opt.test_file)

    # replace abbr
    entityAbbres = loadAbbreviations(opt.abbre_file)
    preprocessMentions(traindocuments, devdocuments, testdocuments, entityAbbres)

    # load dict
    dict = load_dict(opt.dict_file)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained(opt.bert_dir)
    meshlabels, meshlabel_to_ix = parser_dict(dict)

    dict_instances = getDictInstance(dict, meshlabel_to_ix, tokenizer)
    train_instances = getNormInstance(traindocuments, meshlabel_to_ix, tokenizer)
    dev_instances = getNormInstance(devdocuments, meshlabel_to_ix, tokenizer)
    test_instances = getNormInstance(testdocuments, meshlabel_to_ix, tokenizer)

    logging.info('dict_instances_len {}'.format(len(dict_instances)))
    logging.info('train_instance_len {}'.format(len(train_instances)))

    dict_loader = DataLoader(dict_instances, opt.batch_size, shuffle=True, collate_fn=my_collate)
    train_loader = DataLoader(train_instances, opt.batch_size, shuffle=True, collate_fn=my_collate)
    dev_loader = DataLoader(dev_instances, opt.batch_size, shuffle=False, collate_fn=my_collate)
    test_loader = DataLoader(test_instances, opt.batch_size, shuffle=False, collate_fn=my_collate)


    logging.info(opt.gpu)
    logging.info(torch.cuda.is_available())
    if opt.gpu >= 0 and torch.cuda.is_available():
        device = torch.device('cuda', opt.gpu)
    else:
        device = torch.device('cpu')

    # model = BertForSequenceClassification.from_pretrained('bert-base-uncased', cache_dir='/Users/feili/.pytorch_pretrained_bert/distributed_-1',
    #                                                       num_labels = len(meshlabel_to_ix))
    model = BertForSequenceClassification.from_pretrained(opt.bert_dir,
                                                          num_labels = len(meshlabel_to_ix))
    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.learning_rate)

    criterion = nn.CrossEntropyLoss()

    if opt.pretraining:

        dict_iter = iter(dict_loader)
        dict_num_iter = len(dict_loader)

        bad_counter = 0
        best_accuracy = 0

        # start training dictionary
        logging.info("batch_size: %s,  dict_num_iter %s" % (str(opt.batch_size), str(dict_num_iter)))
        # for epoch in range(opt.dict_iteration):
        for epoch in range(opt.pretrain_epoch):
            epoch_start = time.time()

            correct_1, total_1 = 0, 0

            model.train()

            for i in range(dict_num_iter):
                dict_inputs, _, dict_sentences, mask, dict_targets = endless_get_next_batch(dict_loader, dict_iter)
                dict_batch_output = model(dict_inputs, dict_sentences, mask, dict_targets)
                dict_cost = criterion(dict_batch_output, dict_targets)

                total_1 += len(dict_inputs[1])
                _, dict_pred = torch.max(dict_batch_output, 1)

                correct_1 += (dict_pred == dict_targets).sum().item()

                dict_cost.backward()
                optimizer.step()
                model.zero_grad()

            epoch_finish = time.time()
            epoch_cost = epoch_finish - epoch_start

            accuracy = 100.0 * correct_1 / total_1
            logging.info('Epoch {}, time {:.2f}, Dict Training Accuary: {:.2f}%'.format((epoch + 1), epoch_cost, accuracy))

            if accuracy > opt.expected_accuracy:
                logging.info("Exceed expected training accuracy, breaking ... ")
                break

            if accuracy > best_accuracy:
                logging.info("Exceed previous best accuracy: %.2f" % (best_accuracy))
                best_accuracy = accuracy

                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter >= opt.patience:
                logging.info('Pretraining Early Stop!')
                break


    ##train corpus
    train_iter = iter(train_loader)
    num_iter = len(train_loader)


    best_dev_f = -10

    bad_counter = 0

    for epoch in range(opt.max_epoch):
        model.train()
        optimizer.zero_grad()
        # sum_cost = 0.0
        correct, total = 0, 0

        epoch_start = time.time()

        for i in range(num_iter):
            mention_inputs, _, sentences, mask, targets = utils.endless_get_next_batch(train_loader, train_iter)

            batch_output = model(mention_inputs, sentences, mask)
            cost = criterion(batch_output, targets)

            # sum_cost += cost.item()

            # train accuracy
            total += len(mention_inputs[1])
            _, pred = torch.max(batch_output, 1)
            # correct += (pred == targets).sum().sample_data[0]
            correct += (pred == targets).sum().item()

            cost.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start

        logging.info('Epoch {}, time {:.2f}, Training Accuary: {:.2f}%'.format((epoch + 1), epoch_cost, 100.0 * correct / total))

        # evaluate on test sample_data
        test_acc, test_instances = evaluate(test_loader, model, meshlabel_to_ix)
        logging.info('Epoch {}, Testing Accuary: {:.2f}%'.format((epoch + 1), test_acc))

        p, r, f = utils.calculateMacroAveragedFMeasure(test_instances, testdocuments)
        logging.info('Epoch {}, Macro P= {:.4f}, R= {:.4f}, F= {:.4f}'.format((epoch + 1), p, r, f))

        if f > best_dev_f:
            logging.info("Exceed previous best f score on test: %.4f" % (best_dev_f))

            best_dev_f = f

            bad_counter = 0

        else:
            bad_counter += 1

        if bad_counter >= opt.patience:
            logging.info('Early Stop!')
            break

    logging.info('norm end')


