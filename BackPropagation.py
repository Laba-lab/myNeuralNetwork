# Trying to use input information from Building_AI Exercise 21 and
# adopt to Backpropagation example of NN from https://dev.to/shamdasani/buiild-a-flexible.... 

# Exercise 21 network consits of 5 input nodes, a hidden layer with two nodes, 
# second hidden layer with two nodes and finally output node 

import numpy as np

X = np.array([  [111, 13, 12, 1, 161],
                [125, 13, 66, 1, 468],
                [46, 6, 127, 2, 962],
                [80, 9, 80, 2, 816],
                [33, 10, 18, 2, 297],
                [85, 9, 111, 3, 601],
                [24, 10, 105, 2, 1072],
                [31, 4, 66, 1, 417],
                [56, 3, 60, 1, 36],
                [49, 3, 147, 2, 179] ], dtype=float)
                
x_test = np.array([ [82, 2, 65, 3, 516],
                    [72, 2, 25, 3, 450],
                    [60, 3, 15, 1, 300], 
                    [74, 5, 10, 2, 100] ], dtype=float)

y = np.array([[335800.0], [379100.0], [118950.0], [247200.0], [107950.0], [26550.0], [75850.0], [93300.0], [170650.0], [149000.0]], dtype=float)

# bias nodes (as Term Intercept 'a' in Linear regression))
b0 = np.array([-4.21310294, -0.52664488])
b1 = np.array([-4.84067881, -4.53335139])
b2 = np.array([-7.52942418])


# SCALE UNITS
# We want to normalize units as our inputs are in hours, but our output is a test score from 0-100.
# Therefore, we need to scale our data by dividing by the maximum value for each variable
X = X/np.amax(X, axis=0)  # maximum of X array
y = y/1000000.0                # maximum price of the cabin is 379100
b0 = b0/7.52942418
b1 = b1/7.52942418
b2 = b2/7.52942418

# DEFINE a python "class" and write an "init" function where we'll specify our parameters such as 
# input, hidden, and output layers
class Neural_Network(object):
    def __init__(self):
        #parameters
        self.inputSize = 5    # 5 input nodes
        self.outputSize = 1   # one outut node
        self.hidden1Size = 2   # 2 nodes in first hidden layer
        self.hidden2Size = 2   # 2 hidden nodes in second hidden layer
        
        # GENERATE INITIAL WEIGHTS RANDOMLY
        # We need 3 sets of weights, one to go from the input to the 1st hidden layer,
        # another to go from 1st hidden layer to 2nd hidden layer,
        # and other set of weights to go from the 2nd hidden layer to output layer
        self.W0 = np.random.randn(self.inputSize, self.hidden1Size)  # (5x2) weights - Five input nodes to two hidden nodes
        self.W1 = np.random.randn(self.hidden1Size, self.hidden2Size)   # (2x2) weights - 2 hidden to 2 2nd hidden
        self.W2 = np.random.randn(self.hidden2Size, self.outputSize) # (3x1) weights
        
        
    # **** FORWARD PROPAGATION FUNCTION ****
    # Let's pass in our input X and use variable z to simulate the activity between the input and output layers
    # We need to take a dot product (martix multiplication) of the inputs and weights,
    # apply an activation function, take another dot product of the hidden layer and 
    # another set ow weights, and lastly apply a final activation function to recive the output.
    def forward(self, X):
        # forward propagation through our network
        self.z0 = np.dot(X, self.W0)+b0    # dot product of X (input) and first set of weights
        self.z1 = self.sigmoid(self.z0)    # activation function gives output from first hidden layer
        
        self.z2 = np.dot(self.z1, self.W1)+b1    # dot product of 1st hidden layer becomes input to 2nd hidden layer
        self.z3 = self.sigmoid(self.z2)          # activation function gives output from second hidden layer
        
        self.z4 = np.dot(self.z3, self.W2)+b2   # dot product of 2nd hidden layer and weights W2 gives input to output node
        o = self.sigmoid(self.z4)               # activation function for output
        
        return o
        
        
    # DEFINE A SIGMOID FUNCTION
    def sigmoid(self, s):
        return 1/(1+np.exp(-s))
        
     
    # DEFINE SIGMOID PRIME - DERIVATIVE OF SIGMOID FUNCTION    
    def sigmoidPrime(self, s):
        # derivative of sigmoid
        return s*(1-s)
        
        
    # **** DEFINE A BACKWARD PROPAGATION FUNCTION **** 
    # That does everything specified in four steps of 
    # calculating the incremental change to our weights:
    def backward(self, X, y, o):
        # backward propagate through the network
        self.o_error = y - o 
        #print("This is self.o_error = y - o", self.o_error)                             # error in output (STEP#1)
        self.o_delta = self.o_error*self.sigmoidPrime(o)  # apply derivative of sigmoid to error (STEP#2)
        #print(" This is self.o_delta:", self.o_delta)
        
        self.z3_error = self.o_delta.dot(self.W2.T)  # z3 error: how much our second hidden layer weights contributed to output error
        #print("This is self.z3_error:", self.z3_error)
        self.z3_delta = self.z3_error*self.sigmoidPrime(self.z3)  # applying derivative of sigmoid to z2 error
        #print(" This is self.z3_delta:", self.z3_delta)
        
        self.z1_error = self.z3_delta.dot(self.W1.T)  # z1 error: how much our first hidden layer weights contributed to output error
        #print("This is self.z1_error:", self.z1_error)
        self.z1_delta = self.z1_error*self.sigmoidPrime(self.z1)  # applying derivative of sigmoid to z1 error
        #print("This is self.z1_delta:", self.z1_delta)
        
        #self.z3_error = self.o_delta.dot(self.W2.T)  # z2 error: how much our second hidden layer weights contributed to output error
        #self.z3_delta = self.z3_error*self.sigmoidPrime(self.z3)  # applying derivative of sigmoid to z2 error
        
        self.W0 += X.T.dot(self.z1_delta)      # adjusting first set (input --> hidden) weights
        self.W1 += self.z1.T.dot(self.z3_delta)  # adjusting second set of weights hidde1 --> hidden2
        self.W2 += self.z3.T.dot(self.o_delta)   # adjusting second set (hidden2 --> output) weights
   
    
    # DEFINE TRAIN FUNCTION
    # We can now define our output through initiating forward propagation and initiate 
    # the backward function by calling it in the "Train" function
    def train(self, X, y):
        o = self.forward(X)
        self.backward(X, y, o)
       
      
# To run the network, all we have to do is to run the "train" function. Of course, we'll want to do this multiple times, 
# or maybe thousand of times. So we'll use a for loop          
NN = Neural_Network()

for i in range(10000):
    print("Input: \n" + str(X))
    print("Actual output: \n" + str(y))
    print("Predicted output: \n" + str(NN.forward(X)))
    print("Loss: \n" + str(np.mean(np.square(y-NN.forward(X)))))
    print("\n")
    NN.train(X, y)
    
    
# TEST: 10 000 iteracija
# Raspberry3 2 min in 37 sekunda = 157 000 miliseconds
# 15.7 ms po iteraciji

# TEST2: 10 000 iteracija
# Included bias node values
# 2 min 37 s

# TEST3: 10 000 iteracija
# Raspberry 4. Included bias node values
# 45 s. 4.5 ms po iteraciji

# TEST4: 10 000 iteracija
# Google Colab Server
# 44 s, 4.32 ms po iteraciji
