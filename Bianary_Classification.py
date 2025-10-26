import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy
import pathlib
from pathlib import Path
import os
import sklearn
from sklearn.datasets import make_circles #this is for datasets
import pandas as pd
import requests #this is just used to request imports from sitesaka rather than download we just request the functions

#-------------------------------DATA--------------------------------------

n_samples = 1000

X,y = make_circles(n_samples,
                   noise=0.03,
                   random_state=42)
# print(len(X),len(y))

#VISUALISING

# print(f"samples of x: {X[:5]}")
# print(f"samples of y: {y[:5]}")

# circles = pd.DataFrame({"x1":X[:,0],
#                         "x2":X[:,1],
#                         "label":y
# })
# print(circles.head(10))

# plt.scatter(x = X[:,0],
#             y = X[:,1],
#             c=y,
#             cmap=plt.cm.RdYlBu) # type: ignore
# plt.show()

#this data were working with is called a toy dataset as it is smaller compared to actual ones,
#but is big enough to train models and test them reliabely

#CHECKING INPUT OUTPUT SHAPES

"""
this is just to show that 2 features of x is used to predict 1 feature of y
"""
X_samples = X[0]
y_samples = y[0]

print(f"values of sample of x {X_samples} same for y {y_samples}")
print(f"shapes of sample of x {X_samples.shape} same for y {y_samples.shape}")

# TURNING DATA INTO TENSOR

#this is done as default value of numpy is foloat 64 and of torch is float 32 so we do this to prevent any data type errors in the future
X = torch.from_numpy(X).type(torch.float32) 
y = torch.from_numpy(y).type(torch.float32)

#SPLITTING DATA INTO TRAINING AND TEST SETS
from sklearn.model_selection import train_test_split

#this is imported to randomly split data into training and testing

X_train , X_test , y_train , y_test = train_test_split(X,
                                                       y,
                                                       test_size=0.2,  #default size is 0.25
                                                       random_state=42)
#the format of X_trian,X_test,y_train<y_test is predefined in the train_test_split function so make sure to keep it in that order only

print(len(X_train))
print(len(X_test))
print(len(y_train))
print(len(y_test))

#-----------------------------BUILDING THE MODEL--------------------------------

#DEVICE AGNOSTIC CODE
device = "cuda" if torch.cuda.is_available() else "cpu"

# #CREATING CLASS THE NORMAL WAY
# class CircleModelV1(nn.Module):
#     def __init__(self):
#         super().__init__()
#         #create 2 linear layers so its able to handel shape of our data

#         #HIDDEN LAYER
#         """
#         the job of this layer is that it converts the in features aka tensor and upscales it to he amt of out features specified
#         it does this because right now the model only has 2 units of data to find pattern in to find y
#         but if we upscale it to 5 it will have more units of data to find patterns in for each value of y
#         we dont just scale in features arbitrarily as it has point after which its effectiveness starts to deteriorate as it takes more time to process one single y value aka same reason we dont keep lr to 0.0000001
#         we usually keep the out feautres as multiples of 8
#         """

#         self.layer_1 = nn.Linear(in_features=2,
#                                  out_features=5)
        
#         #OUTPUT LAYER
#         """
#         this layers job is to basically change the upscaled data units into a type compatible with our output/prediction
#         aka as we have 5 data units per value of y so we need to downscale it and bring it to 1 so we dont get a error trying to plot 5 things for when what we need to predict is just 1
#         the infeatues of this need to be the same as the output layer of the previous layer to avoid shape erros
#         """
        
#         self.layer_2 = nn.Linear(in_features=5, #in features is equal to the dimention of ur input tensor aka X in this case
#                                  out_features=1)#out features is equal to the number of things we want the model to classify the data into
#         #FORWARD PASS
#     def forward(self,x):
#         return self.layer_2(self.layer_1(x)) #here it just says that x -> layer1 -> layer2
    

#CRATING A MODEL USING 'nn.sequential'
"""
this is just a easier way to make layers of a neural network
it is good for small models but for bigger and complex models preffer making ur own class n stuff to know whats going on beneth the covers
"""
model_0 = nn.Sequential(
    nn.Linear(in_features=2,out_features=5),
    nn.Linear(in_features = 5,out_features=10),
    nn.Linear(in_features=10,out_features=1)
).to(device)


#CREATING INSTANCE OF OUR CLASS
# model_0 = CircleModelV1().to(device)
# print(model_0,next(model_0.parameters()).device)

# print(model_0.state_dict()) 
#hear we can see that the o.weight is 10 as 2*5=10 aka in features *out features
#the 0.bias is 5 tensors as we said it to convert to 5
#same with the 1.weight as it takes input from the 0.bias its 5
#the 1.bias is 1 tensor as we said it to convert the 5 inputs to 1 output


#MAKING PREDICTIONS
with torch.inference_mode():
    untrained_preds = model_0(X_test.to(device))
print(f"length of preds {len(untrained_preds)} shape {untrained_preds.shape}")
print(f"length of test samples {len(X_test)} shape {X_test.shape}")
print(f"\nfirst 10 preds {untrained_preds[:10]}")
print(f"first 10 labels {y_test[:10]}")

#SETTING LOSS FUNCTION AND OPTIMIZER

"""
the loss function to use is problem specific ex,
1. for regression we use MAE or MSE (mean absolute error and mean squared error)
2. for classification we use binary cross entropy or cateforical cross entropy
 REFFER FOR CROSS ENTROPY:-  https://medium.com/data-science/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
"""

#LOSS FUNCTION
#we will use torch.nn.BCEWithLogitsLoss()

# loss_fn = nn.BCELoss()#here the inputs have to go through the sigmod activation function frior to this
loss_fn = nn.BCEWithLogitsLoss()#here the inputs dont have to go through the sigmoid activation functionbefore entering as it is builtin
                                #this also is more numerically stable

"""
WHAT IS ACTIVATION FUNCTION:
activation function is applied to the output of a neuron
it basically says if the output of neuron is more than 0 pass else it blocks it
by doing these stuff it introduces non linearity

WHAT IS LOGIT:
logit is the output of the neuron befor it passes through the activation function

"""

optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

#calulating accuracy
def accuracy_fn(y_true,y_pred):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct/len(y_pred))*100
    return acc

#GOING FROM LOGITS -> PREDICTION PROPABILITIES -> PREDICTION LABLES
#this is to make predictiosn same format as test labels

# #LOGITS
# model_0.eval()
# with torch.inference_mode():
#     y_logits = model_0(X_test.to(device))[:5]
# print(y_logits)

# #SIGMOID
# y_pred_probs = torch.sigmoid(y_logits)
# print(y_pred_probs)#as we can see the sigmoid function has filtered out all the negetive values and we are only left with positive values 

# #PREDICTING LABLES
# #this rounds the values of the sigmoid and sees what the output is truing to predict if its a 0 or a 1
# y_preds = torch.round(y_pred_probs)

# #in full
# #to compress all the above code into one line
# y_preds_labels = torch.round(torch.sigmoid(model_0(X_test.to(device))[:5]))


# #checking for equality
# print(torch.eq(y_preds.squeeze(),y_preds_labels.squeeze()))
# print(y_preds.squeeze())

# #---------------------------------TRAINING LOSS------------------------------
# #steps:-
# """
# 1. forward pass
# 2. calculate loss
# 3.optimizer zero grad
# 4.back propagation
# 5.optimizer step (grad decent)
# """

# torch.cuda.manual_seed(42)

# epochs = 2000

# X_train,y_train = X_train.to(device),y_train.to(device)
# X_test,y_test = X_test.to(device),y_test.to(device)

# for epoch in range(epochs):
#     model_0.train()

#     #forward pass
#     y_logits = model_0(X_train)

#     y_pred = torch.round(torch.sigmoid(y_logits))

#     #calculate loss
#     # loss = loss_fn(torch.sigmoid(y_logits),   #this is for BCEloss
#     #                y_train)

#     loss = loss_fn(y_logits.squeeze(),  #this is for BCEWithLogitLoss as it expects raw logits
#                    y_train)   #here we have to enter in the way of predictions and then true labels
    
#     acc = accuracy_fn(y_true = y_train,
#                       y_pred = y_pred)
    
#     #optimizer zero grad
#     optimizer.zero_grad()

#     #loss backward
#     loss.backward()

#     #optimizer step
#     optimizer.step()

#     #TESTING
#     model_0.eval()
#     with torch.inference_mode():
#         test_logits = model_0(X_test)
#         test_pred = torch.round(torch.sigmoid(test_logits))

#         test_loss = loss_fn(test_logits.squeeze(),
#                             y_test)
        
#         test_acc = accuracy_fn(y_true=y_test,    #here we do true labels first then preds as scikitlearn has it like that
#                                y_pred=test_pred)
        
#         #printing whats happening
#         if epoch % 100 == 0:
#             print(f"epoch {epoch} | loss{loss.item():.5f} , acc: {acc:.2f}%| test loss {test_loss:.5f} test acc {acc:.2f}%")

#making predictions and evaluate model

#importing helper functions
"""
these are just premade funtions to make it easier to do some stuff
we imported the library "requests" for this
"""

if Path("helper_functions.py").is_file():
    print("helper functions already exists")
else:
    print("downloading helper function.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/refs/heads/main/helper_functions.py")
    with open("helper_functions.py","wb") as f:
        f.write(request.content)
from helper_functions import plot_predictions,plot_decision_boundary

# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)
# plt.title("Train")
# plot_decision_boundary(model_0,X_train,y_train)
# plt.subplot(1,2,2)
# plt.title("Test")
# plot_decision_boundary(model_0,X_train,y_train)
# plt.show()

"""
here the accuracy is only at 50%accuracy as were only using straight line to divide circular data 
as seen im the diagram 
"""
#-------------------------------------IMPROVING THE MODEL--------------------------------------

"""
WAYS TO IMPORVE OUT MODEL:-

1. add more layers - adding more layers gives the model more data to find patterns in
2. add more hidden units - adding more hidden units also gives the model more data to find patterns in
3. run for longer - running it for longer gives it more time to find patterns
4. changing the activation functions - applying diff activaion functions so that it uses diff formula to change logits into final ans
5. change the learning rate - this will just change the minimum the model will change the params
6. change the loss function 

these are options to improve the model from the models prespective aka changing the model not the data to improve effectiveness

and these are called hyper parameters as these are stuff we can change and are in our control
"""

class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2,out_features=10)
        self.layer_2 = nn.Linear(in_features=10 , out_features=10)
        self.layer_3 = nn.Linear(in_features=10,out_features=10)
        self.layer_4 = nn.Linear(in_features=10,out_features=1)
        self.relu = nn.ReLU() #this just turns -ve inuts to 0 and leaves +ve inputs alone and hence introduces non lineaity
    def forward(self,t):   #does not matter what u put in this as its just a placeholder for hatever we put in
        # z = self.layer_1(t)
        # z = self.relu(z)    #adding non linear functions
        # z = self.layer_2(z)
        # z = self.relu(z)
        # z = self.layer_3(z)
        #compressing th above into a single line
        return self.layer_4(self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(t))))) )
        #this method also leverages speedups wherever possible
model_1 = CircleModelV2().to(device)
print(model_1,model_1.state_dict())

#for loss function  lests just use the one we created previously

#for optimizer we will need to create new as we will have to pass the new models params

optimizer1 = torch.optim.SGD(params=model_1.parameters(),lr = 0.1) 

torch.manual_seed(42)
torch.cuda.manual_seed(42)

#THIS IS JUST FOR TRAINING AS WEVE DEFINED SOME OF THE STUFF ABOVE


epochs = 1500
#sending data to gpu
X_train,y_train = X_train.to(device),y_train.to(device)
X_test,y_test = X_test.to(device),y_test.to(device)


for epoch in range(epochs):
    model_1.train()

    y_logits = model_1(X_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,y_train)
    acc = accuracy_fn(y_true=y_train,
                      y_pred=y_pred)
    
    optimizer1.zero_grad()

    loss.backward()

    optimizer1.step()

    model_1.eval()
    with torch.inference_mode():
        test_logits = model_1(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))

        test_loss = loss_fn(test_logits,
                            y_test)
        
        test_acc = accuracy_fn(y_true=y_test,
                               y_pred=test_pred)
        

    if epoch % 100 == 0:
        print(f"epoch {epoch} | loss {loss:.5f} | acc {acc:.2f} | test loss {test_loss:.5f} | test acc {test_acc:.2f}%")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_1,X_train,y_train)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model_1,X_train,y_train)
plt.show()


# #--------------------------------REPLICATIONG NON-LINEAR ACTIVATION FUNCTIONS------------------------------------------

# A = torch.arange(-10,10,0.5,dtype=torch.float32)

# print(A)
# #RELU

# # plt.plot(torch.relu(A))

# def Rrelu(x):
#     return torch.max(torch.tensor(0),x)
# plt.plot(Rrelu(A))

# #SIGMOID

# def Rsigmoid(x):
#     return 1/(1+torch.exp(-x))
# plt.plot(Rsigmoid(A))
# plt.show()


