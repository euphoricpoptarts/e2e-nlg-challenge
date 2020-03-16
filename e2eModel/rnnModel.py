import torch
from torch import nn

class RNNModel(nn.Module):
    """description of class"""
    def __init__(self, mr_vocab, sen_vocab, device):
        self.device = device
        self.mr_word2index = dict(map(reversed, enumerate(mr_vocab)))
        self.mr_index2word = dict(enumerate(mr_vocab))
        self.sen_word2index = dict(map(reversed, enumerate(sen_vocab)))
        self.sen_index2word = dict(enumerate(sen_vocab))
        self.sen_offset = len(mr_vocab)
        self.vocab_size = len(mr_vocab) + len(sen_vocab)

        super(RNNModel, self).__init__()
        self.hidden_dim = 100
        self.embed_dim = 100
        self.embed = nn.Embedding(self.vocab_size, self.embed_dim)
        self.rnn = nn.GRU(self.embed_dim, self.hidden_dim, 10, batch_first=True)
        self.outL = nn.Linear(self.hidden_dim, self.vocab_size)

    def onehot(self, dataTuple):
        mr_idx = [self.mr_word2index[w] for w in dataTuple[0]]
        sen_idx = [self.sen_offset + self.sen_word2index[w] for w in dataTuple[1]]
        #encoding = torch.zeros([1, len(mr_idx) + len(sen_idx), self.vocab_size])
        #seq_offset = len(mr_idx)
        #for seq, voc in enumerate(mr_idx):
        #    encoding[0, seq, voc] = 1
        #for seq, voc in enumerate(sen_idx):
        #    encoding[0, seq + seq_offset, voc] = 1
        target = mr_idx + sen_idx
        l = len(target)
        encoding = torch.zeros([1, l - 1])
        encoding[0] = torch.Tensor(target[:-1])
        return encoding, torch.Tensor(target[1:])

    def getTopWords(self, input, B):
        #-1 index for last output
        prob = nn.functional.softmax(input[-1], dim=0)

        # Taking the B classes with the highest probability scores from the output
        idx = torch.topk(prob[self.sen_offset:], k=B).indices.tolist()
        topProbs = [prob[i].item() for i in idx]
        topWords = [self.sen_index2word[i] for i in idx]
        return list(zip(topWords, topProbs))

    def forward(self, x):
        
        batch_size = x.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        embeds = self.embed(x.long())
        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(embeds, hidden)
        
        # Reshaping the outputs such that it can be fit into the fully connected layer
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.outL(out)
        
        return out, hidden
    
    def init_hidden(self, batch_size):
        # This method generates the first hidden state of zeros which we'll use in the forward pass
        # We'll send the tensor holding the hidden state to the device we specified earlier as well
        hidden = torch.randn(10, batch_size, self.hidden_dim).to(self.device)
        return hidden