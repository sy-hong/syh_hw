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
import string
from argparse import ArgumentParser
import pickle

unk = '<UNK>'
# Consult the PyTorch documentation for information on the functions used below:
# https://pytorch.org/docs/stable/torch.html
class RNN(nn.Module):
    def __init__(self, input_dim, h):  # Add relevant parameters
        super(RNN, self).__init__()
        self.h = h                      # hidden state
        self.numOfLayer = 1             # num layer
        self.rnn = nn.RNN(input_dim, h, self.numOfLayer, nonlinearity='tanh') # RNN
        self.W = nn.Linear(h, 5)        # linear layer
        self.softmax = nn.LogSoftmax(dim=-1) # softmax on the 1st dim; converts outputs -> probability distr. over 5 classes
        self.loss = nn.NLLLoss()        # type of loss


    def compute_Loss(self, predicted_vector, gold_label):
        return self.loss(predicted_vector, gold_label)


    def forward(self, inputs):
        # [to fill] obtain hidden layer representation (https://pytorch.org/docs/stable/generated/torch.nn.RNN.html)
        h_0 = torch.zeros(self.numOfLayer, inputs.size(1), self.h) # explicitly initialize h_0 -- optional based on the documentation
        output, hidden = self.rnn(inputs, h_0)                     # pass inputs (seq length, batch, hidden) & h_0 to the RNN layer
        # print(">> output: ", output.size())                        # output:   (sample) torch.Size([29, 1, 16])  - torch.Size([output features, batch, class #])
        # print(">> hidden: ", hidden.size())                        # hidden:   torch.Size([1, 1, hidden]) - torch.Size([final hidden state, batch, hidden])

        # [to fill] obtain output layer representations 
        output_layer = self.W(output)                              # perform the linear transformation on the final hidden state 
        # print(">> output_layer: ", output_layer.size())            # output_layer:  torch.Size([29, 1, 5]) - torch.Size([output features, batch, class #])

        # [to fill] sum over output 
        # print(">> output_layer.sum(0): ", output_layer.sum(0).size())  # torch.Size([1, 5]) - sum over the layer dimension 
        sum_output = output_layer.sum(0).squeeze(0)               # squeeze the batch dimension out --> need to change line 26's dim from 1 to -1
        # print(">> sum_output: ", sum_output.size())               # sum_output:  torch.Size([5]) - torch.Size([class #])

        # [to fill] obtain probability dist.
        predicted_vector = self.softmax(sum_output)               # pass sum_output to the softmax to get probability distribution
        # print(">> predicted_vector: ", predicted_vector.size())   # predicted_vector:  torch.Size([5]) - torch.Size([class #])

        return predicted_vector


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

    # write out to results-rnn/test.out
    result_dir = 'results-rnn'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    val_loss = []
    train_loss = []
    train_acc = []
    val_acc = []

    with open(os.path.join(result_dir, "output.out"), "a") as f:
        def log(logging):
            print(logging)
            f.write(logging + "\n")

        log("========== RNN for the hidden dimension of {} ==========".format(args.hidden_dim))

        log("========== Loading data ==========")
        train_data, valid_data = load_data(args.train_data, args.val_data) # X_data is a list of pairs (document, y); y in {0,1,2,3,4}

        # Think about the type of function that an RNN describes. To apply it, you will need to convert the text data into vector representations.
        # Further, think about where the vectors will come from. There are 3 reasonable choices:
        # 1) Randomly assign the input to vectors and learn better embeddings during training; see the PyTorch documentation for guidance
        # 2) Assign the input to vectors using pretrained word embeddings. We recommend any of {Word2Vec, GloVe, FastText}. Then, you do not train/update these embeddings.
        # 3) You do the same as 2) but you train (this is called fine-tuning) the pretrained embeddings further.
        # Option 3 will be the most time consuming, so we do not recommend starting with this

        log("========== Vectorizing data ==========")
        model = RNN(50, args.hidden_dim)  # Fill in parameters
        # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        optimizer = optim.Adam(model.parameters(), lr=0.01) 
        word_embedding = pickle.load(open('./word_embedding.pkl', 'rb'))

        stopping_condition = False
        epoch = 0

        last_train_accuracy = 0
        last_validation_accuracy = 0

        while not stopping_condition:
            random.shuffle(train_data)
            model.train()
            # You will need further code to operationalize training, ffnn.py may be helpful
            log("Training started for epoch {}".format(epoch + 1))
            train_data = train_data
            correct = 0
            total = 0
            minibatch_size = 16
            N = len(train_data)

            loss_total = 0
            loss_count = 0
            for minibatch_index in tqdm(range(N // minibatch_size)):
                optimizer.zero_grad()
                loss = None
                for example_index in range(minibatch_size):
                    input_words, gold_label = train_data[minibatch_index * minibatch_size + example_index]
                    input_words = " ".join(input_words)

                    # Remove punctuation
                    input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()

                    # Look up word embedding dictionary
                    vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i in input_words]

                    # Transform the input into required shape
                    vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                    output = model(vectors)

                    # Get loss
                    example_loss = model.compute_Loss(output.view(1,-1), torch.tensor([gold_label]))

                    # Get predicted label
                    predicted_label = torch.argmax(output)

                    correct += int(predicted_label == gold_label)
                    # print(predicted_label, gold_label)
                    total += 1
                    if loss is None:
                        loss = example_loss
                    else:
                        loss += example_loss

                loss = loss / minibatch_size
                loss_total += loss.data
                loss_count += 1
                loss.backward()
                optimizer.step()

            print(loss_total/loss_count)
            log("Training completed for epoch {}".format(epoch + 1))
            log("Training accuracy for epoch {}: {}".format(epoch + 1, correct / total))
            log("Training LossP {}".format(loss))

            train_loss.append(str(loss))
            train_loss.append(str(float(loss.item())))
            train_acc.append(str(correct / total))
            trainning_accuracy = correct/total


            model.eval()
            correct = 0
            total = 0
            random.shuffle(valid_data)
            log("Validation started for epoch {}".format(epoch + 1))
            valid_data = valid_data

            for input_words, gold_label in tqdm(valid_data):
                input_words = " ".join(input_words)
                input_words = input_words.translate(input_words.maketrans("", "", string.punctuation)).split()
                vectors = [word_embedding[i.lower()] if i.lower() in word_embedding.keys() else word_embedding['unk'] for i
                        in input_words]

                vectors = torch.tensor(vectors).view(len(vectors), 1, -1)
                output = model(vectors)
                predicted_label = torch.argmax(output)
                correct += int(predicted_label == gold_label)
                total += 1
                # print(predicted_label, gold_label)
            log("Validation completed for epoch {}".format(epoch + 1))