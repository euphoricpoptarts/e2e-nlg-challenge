import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, deque
from rnnModel import RNNModel
import torch
from torch import nn
from math import ceil
import numpy as np

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

#MR is always in quotes, sentence is sometimes in quotes
line_pattern = re.compile('^"(?P<mr>[^"]*)","?(?P<sen>[^"\\n]+)"?')

mr_pattern = re.compile("(?P<prop>[a-zA-Z]+)\\[(?P<val>[^\\]]+)\\]")

name_token = "n4m3_t0k3n"
near_token = "n34r_t0k3n"
start_token = "5t4rt_t0k3n"
end_token = "3nd_t0k3n"
sen_end_token = "<end>"

def preprocess(file):
    """
    Prepare the data and the vocab for the models.
    For expediency, the vocabulary will be all the words
    in the dataset (not split into training/test), so
    the assignment can avoid the OOV problem.
    """
    mr_vocab = set()
    sen_vocab = set()
    mr_vocab.add(start_token)
    sen_vocab.add(end_token)
    dataset = []
    with open(file, 'r') as fr:
        for idx, line in enumerate(fr):
            match = line_pattern.search(line)
            if match:
                groups = match.groupdict()
                mr = groups['mr']
                pairs = mr_pattern.findall(mr)
                sen = groups['sen']
                mr_dict = {}
                mr_sen = ""
                for attr, val in pairs:
                    mr_dict[attr] = val
                    if attr == 'name':
                        mr_sen += attr + " "
                    elif attr == 'near':
                        mr_sen += attr + " "
                    else:
                        mr_sen += attr + " " + val + " "
                mr_sen = mr_sen.strip()
                if 'name' in mr_dict:
                    sen = sen.replace(mr_dict['name'], name_token)
                if 'near' in mr_dict:
                    sen = sen.replace(mr_dict['near'], near_token)
                mr_list = word_tokenize(mr_sen)
                sen_list = word_tokenize(sen)
                sen_list.append(sen_end_token)
                mr_vocab.update(set(mr_list))
                sen_vocab.update(set(sen_list))
                dataset.append((mr_list, sen_list))
    max_mr_len = max(map(lambda x: len(x[0]), dataset))
    max_sen_len = max(map(lambda x: len(x[1]), dataset))
    print(max_mr_len)
    print(max_sen_len)

    padded_dataset = []
    for mr, sen in dataset:
        mr_len = len(mr)
        mr = deque(mr)
        mr.extendleft([start_token]*(max_mr_len - mr_len))
        mr = list(mr)
        sen_len = len(sen)
        sen.extend([end_token]*(max_sen_len - sen_len))
        padded_dataset.append((mr, sen))

    return padded_dataset, list(mr_vocab), list(sen_vocab)

def batch_data(prebatched, minibatch):
    batched = []
    inputs = [x[0] for x in prebatched]
    targets = [x[1] for x in prebatched]
    size = len(prebatched)
    for i in range(0, size, minibatch):
        end_idx = min(size, i + minibatch)
        input_batch = torch.cat(inputs[i:end_idx], 0)
        target_batch = torch.cat(targets[i:end_idx], 0)
        batched.append((input_batch, target_batch))
    return batched

def predictWord(model, mr_list, sen_list):
    
    input, _ = model.onehot((mr_list, sen_list))

    input = input.to("cuda:0")
    
    out, hidden = model(input)

    return model.getWord(out)

def predictSentence(model, mr_list):
    sen = []
    for i in range(20):
        sen.append(predictWord(model, mr_list, sen))
    return ' '.join(sen)

def train(dataset, mr_vocab, sen_vocab):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    model = RNNModel(mr_vocab, sen_vocab, device).to(device)
    
    # Define hyperparameters
    n_epochs = 100
    lr=0.0001

    # Define Loss, Optimizer
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    prebatched = list(map(model.onehot, dataset))
    np.random.shuffle(prebatched)
    minibatch = 32

    batched = batch_data(prebatched, minibatch)
    batched_size = len(batched)
    train_size = int(ceil(batched_size*0.8))

    train_batch = batched[0:train_size]
    test_batch = batched[train_size:batched_size]

    #save memory
    prebatched.clear()
    batched.clear()

    for i in range(n_epochs):
        total_train_loss = 0
        total_test_loss = 0
        for input, target in train_batch:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output, hidden = model(input)
            loss = loss_f(output, target.long())
            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for input, target in test_batch:
                input = input.to(device)
                target = target.to(device)
                output, hidden = model(input)
                loss = loss_f(output, target.long())
                total_test_loss += loss.item()
        print("Epoch: {}, Training Loss: {}, Testing Loss: {}".format(i, total_train_loss, total_test_loss))
        if total_train_loss < 250:
            for p in optimizer.param_groups:
                p['lr'] = 0.00005
        elif total_train_loss < 100:
            for p in optimizer.param_groups:
                p['lr'] = 0.000025
    return model