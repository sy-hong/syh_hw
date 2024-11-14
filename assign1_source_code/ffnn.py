import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import math
import random
import os
import time
from tqdm import tqdm
import json
from argparse import ArgumentParser


unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class FFNN(nn.Module):
    def __init__(self, input_dim, h):
        super(FFNN, self).__init__()
        self.h = h
        self.W1 = nn.Linear(input_dim, h)
        self.activation = nn.ReLU()    # The rectified linear unit; one valid choice of activation function
        self.output_dim = 5
        self.W2 = nn.Linear(h, self.output_dim)

        self.softmax = nn.LogSoftmax() # The softmax function that converts vectors into probability distributions; computes log probabilities for computational benefits
        self.loss = nn.NLLLoss()       # The cross-entropy/negative log likelihood loss taught in class

    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)

    def forward(self, input_vector):
        # [to fill] obtain first hidden layer representation
        hidden = self.W1(input_vector)          # pass through 1st linear layer
        hidden = self.activation(hidden)        # pass the hidden layer through the activation function
        
        # [to fill] obtain output layer representation
        output_layer = self.W2(hidden)          # pass through the 2nd linear layer
        
        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(output_layer)  # pass through softmax
        
        return predicted_vector


# Returns: 
# vocab = A set of strings corresponding to the vocabulary
def make_vocab(data):
    vocab = set()
    for document, _ in data:
        for word in document:
            vocab.add(word)
    return vocab 


# Returns:
# vocab = A set of strings corresponding to the vocabulary including <UNK>
# word2index = A dictionary mapping word/token to its index (a number in 0, ..., V - 1)
# index2word = A dictionary inverting the mapping of word2index
def make_indices(vocab):
    vocab_list = sorted(vocab)
    vocab_list.append(unk)
    word2index = {}
    index2word = {}
    for index, word in enumerate(vocab_list):
        word2index[word] = index 
        index2word[index] = word 
    vocab.add(unk)

    return vocab, word2index, index2word 


# Returns:
# vectorized_data = A list of pairs (vector representation of input, y)
def convert_to_vector_representation(data, word2index):
    vectorized_data = []
    for document, y in data:
        vector = torch.zeros(len(word2index)) 
        for word in document:
            index = word2index.get(word, word2index[unk])
            vector[index] += 1
        vectorized_data.append((vector, y))

    return vectorized_data



def load_data(train_data, val_data):
    with open(train_data) as training_f:
        training = json.load(training_f)
    with open(val_data) as valid_f:
        validation = json.load(valid_f)

    tra = []
    val = []
    for elt in training:
        tra.append((elt["text"].split(),int(elt["stars"]-1)))
    for elt in validation:
        val.append((elt["text"].split(),int(elt["stars"]-1)))

    return tra, val


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-hd", "--hidden_dim", type=int, required = True, help = "hidden_dim")
    parser.add_argument("-e", "--epochs", type=int, required = True, help = "num of epochs to train")
    parser.add_argument("--train_data", required = True, help = "path to training data")
    parser.add_argument("--val_data", required = True, help = "path to validation data")
    parser.add_argument("--test_data", default = "to fill", help = "path to test data")
    parser.add_argument('--do_train', action='store_true')
    args = parser.parse_args()

    # fix random seeds
    random.seed(42)
    torch.manual_seed(42)

    # write out to results/test.out
    result_dir = 'results-ffnn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    train_loss = []
    train_acc = []
    train_time = []

    val_loss = []
    val_acc = []
    val_time = []

    with open(os.path.join(result_dir, "output.out"), "a") as f:
        def log(logging):
            print(logging)
            f.write(logging + "\n")

        log("** FFNN for the hidden dimension of {} **".format(args.hidden_dim))
        log("** Total epochs {} **".format(args.epochs))
        # load data 
        log("========== Loading data ==========")
        train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}
        vocab = make_vocab(train_data)
        vocab, word2index, index2word = make_indices(vocab)

        log("========== Vectorizing data ==========")
        train_data = convert_to_vector_representation(train_data, word2index)
        valid_data = convert_to_vector_representation(valid_data, word2index)
        

        model = FFNN(input_dim = len(vocab), h = args.hidden_dim)
        optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0.9)
        
        log("========== Training for {} epochs ==========".format(args.epochs))
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            log("Training started for epoch {}".format(epoch + 1))
            random.shuffle(train_data) # Good practice to shuffle order of training data
            minibatch_size = 16 
            N = len(train_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
                loss.backward()
                optimizer.step()
            log("Training completed for epoch {}".format(epoch + 1))
            log("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            log("Training time for this epoch: {}".format(time.time() - start_time))
            
            train_loss.append(str(float(loss)))
            train_acc.append(str(correct / total))
            train_time.append(str(time.time() - start_time))

            loss = None
            correct = 0
            total = 0
            start_time = time.time()
            log("Validation started for epoch {}".format(epoch + 1))
            minibatch_size = 16 
            N = len(valid_data) 
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_vector, gold_label = valid_data[minibatch_index * minibatch_size + example_index]
                    predicted_vector = model(input_vector)
                    predicted_label = torch.argmax(predicted_vector)
                    correct += int(predicted_label == gold_label)
                    total += 1
                    example_loss = model.compute_Loss(predicted_vector.view(1,-1), torch.tensor([gold_label]))
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss
                loss = loss / minibatch_size
            log("Validation completed for epoch {}".format(epoch + 1))
            log("Validation accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            log("Validation time for this epoch: {}".format(time.time() - start_time))
            val_acc.append(str(correct / total))
            val_time.append(str(time.time() - start_time))
            val_loss.append(str(float(loss)))
            

        val_acc_epoch = ', '.join(val_acc)
        train_loss_epoch = ', '.join(train_loss)
        val_loss_epoch = ', '.join(val_loss)

        train_acc_epoch = ', '.join(train_acc)
        train_time_epoch = ', '.join(train_time)
        val_time_epoch = ', '.join(val_time)



        log("val_acc_epoch {}".format(val_acc_epoch))
        log("train_loss_epoch {}".format(train_loss_epoch))
        log("val_loss_epoch {}".format(val_loss_epoch))

        log("train_acc_epoch {}".format(train_acc_epoch))
        log("train_time_epoch {}".format(train_time_epoch))
        log("val_time_epoch {}".format(val_time_epoch))
        
        log("========== Complete ==========")
        log("==============================")
        log("")
    
    