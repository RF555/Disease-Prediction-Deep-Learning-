import torch
import torch.nn as nn
from PrepTestData import getTrainAndTest, trainModel, accuracyTest
from torch.utils.data import DataLoader
import torch.nn.functional as F
import math


# Multiclass problem
class NeuralNet(nn.Module):
    def __init__(self, input_size_, hidden_size_1, hidden_size_2, output_size_):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size_, hidden_size_1)
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.linear3 = nn.Linear(hidden_size_2, output_size_)

    def forward(self, x):
        out = F.relu(self.linear1(x))
        out = F.relu(self.linear2(out))
        out = self.linear3(out)
        return out


train, test = getTrainAndTest()

train_size, features_size = train.x_data.shape
print("\ntrain size: ", train_size)
print("features size: ", features_size)
input_size = features_size  # len(train)
print("input size: ", input_size)
output_size = len(train.y_data[0])
print("output size: ", output_size)
print("\n\n")

batch_size = 32
num_epochs = 100
num_iterations = math.ceil(train_size / batch_size)
learning_rate = 0.0001
print_rate = num_epochs / 10

train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

model = NeuralNet(input_size_=input_size,
                  hidden_size_1=64,
                  hidden_size_2=32,
                  output_size_=output_size)

criterion = nn.CrossEntropyLoss()  # applies Softmax
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

trainModel(train_loader, model, criterion, optimizer, num_iterations, print_rate)

test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

print(f'Accuracy of the network on the test set: {accuracyTest(model, test_loader)} %')
print(f'Accuracy of the network on the train set: {accuracyTest(model, train_loader)} %')
