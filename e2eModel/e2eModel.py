import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter, deque
from rnnModel import RNNModel
import torch
from torch import nn

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
    lr=0.01

    # Define Loss, Optimizer
    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    prebatched = list(map(model.onehot, dataset))
    size = len(prebatched)
    print(size)
    minibatch = 10

    batched = []
    inputs = [x[0] for x in prebatched]
    targets = [x[1] for x in prebatched]
    for i in range(0, size, minibatch):
        end_idx = min(size, i + minibatch)
        input_batch = torch.cat(inputs[i:end_idx], 0)
        target_batch = torch.cat(targets[i:end_idx], 0)
        batched.append((input_batch, target_batch))

    #save memory
    inputs.clear()
    targets.clear()
    prebatched.clear()

    for i in range(n_epochs):
        total_loss = 0
        for input, target in batched:
            input = input.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output, hidden = model(input)
            loss = loss_f(output, target.long())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        print("Epoch: {}, Loss: {}".format(i, total_loss))