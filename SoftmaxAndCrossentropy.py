import torch
import torch.nn as nn
from PrepTestData import getTrainAndTest, trainModel, accuracyTest
from torch.utils.data import DataLoader
import math

train, test = getTrainAndTest()
# print("train.x_data.shape:\n", train.x_data.shape)
# print("train.y_data.shape:\n", train.y_data.shape)
# print("test.x_data.shape:\n", test.x_data.shape)
# print("test.y_data.shape:\n", test.y_data.shape)

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

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=train,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

model = nn.Linear(input_size, output_size)

criterion = nn.CrossEntropyLoss()  # applies Softmax
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

trainModel(train_loader, model, criterion, optimizer, num_iterations, print_rate)

test_loader = DataLoader(dataset=test,
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=0)

print(f'Accuracy of the network on the test set: {accuracyTest(model, test_loader)} %')
print(f'Accuracy of the network on the train set: {accuracyTest(model, train_loader)} %')
