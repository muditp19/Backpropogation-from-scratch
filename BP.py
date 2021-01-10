# importing the necessary libraries
import numpy as np 
np.random.seed(42)
# seeding has a big role to play in the convergence of the algorithm.


# defining the sigmoid function for activation mapper
def sigmoid (x):
    return 1/(1 + np.exp(-x))


# defining the derivative to be used for back propogation 
def sigmoid_backward(x):
    return x * (1 - x)

#Input datasets given in problem set 2
input_array = np.array([[1,0],[0,1],[-1,0],[0,-1],[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5]])
target_array = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])

# number of time the data will go through 
epochs = 20000
# define the learning rate here 
l_rate = 0.1
# Mention the number of neurons/units in each layer 
input_Layer, hidden_Layer, output_Layer = 2,4,1 

#Random weights and bias initialization using numpy random uniform distribution
hidden_weights = np.random.uniform(size=(input_Layer,hidden_Layer))
hidden_bias =np.random.uniform(size=(1,hidden_Layer))
output_weights = np.random.uniform(size=(hidden_Layer,output_Layer))
output_bias = np.random.uniform(size=(1,output_Layer))

# looking at the initial weights and biases produced by the random uniform distribution
print("Start hidden weights:\n ",hidden_weights)
print("Start hidden biases:\n ",hidden_bias)
print("Start output weights:\n ",output_weights)
print("Start output biases:\n ",output_bias)



#Training algorithm to repeat for the number of epochs mentioned above
for i in range(epochs):
	#Forward Propagation of the network
	hidlayer_activation = np.dot(input_array,hidden_weights)
	hidlayer_activation = hidlayer_activation + hidden_bias
	hidlayer_output = sigmoid(hidlayer_activation)

	outlayer_activation = np.dot(hidlayer_output,output_weights)
	outlayer_activation = outlayer_activation + output_bias
	pred_output = sigmoid(outlayer_activation)

	#Backpropagation for the network
	e = target_array - pred_output
	d_pred_output = e * sigmoid_backward(pred_output)
        
	error_hidden_layer = d_pred_output.dot(output_weights.T)
	d_hidden_layer = error_hidden_layer * sigmoid_backward(hidlayer_output)

	#Updating Weights and Biases after back propogation
	output_weights = output_weights + hidlayer_output.T.dot(d_pred_output) * l_rate
	output_bias = output_bias + np.sum(d_pred_output,axis=0,keepdims=True) * l_rate
	hidden_weights = hidden_weights + input_array.T.dot(d_hidden_layer) * l_rate
	hidden_bias = hidden_bias + np.sum(d_hidden_layer,axis=0,keepdims=True) * l_rate
    
	if i%1000==0:
		print("epochs :",i)
		print('Error',abs(np.mean(e)))
   
    

# looking at the final weights and biases after the training algo 
print("Final hidden weights: ",hidden_weights)
print("Final hidden bias: ",hidden_bias)
print("Final output weights: ",output_weights)
print("Final output bias: ",output_bias)

# performance of the training lagorithm on the intial input et of data and targets
print("\nOutput from nn after triaining 10,000 epochs: \n",pred_output)


#predict output using the trained weights and new input for a number unseen sets to look for generalization 
test_inputs_1 = np.array([[2,0],[0,2],[-2,0],[0,-2],[1.9,1.9],[-1.9,1.9],[1.9,-1.9],[-1.9,-1.9]])
test_out_1 = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])


test_inputs_2 = np.array([[2,0],[0,2],[-2,0],[0,-2],[1.5,1.5],[-1.5,1.5],[1.5,-1.5],[-1.5,-1.5]])
test_out_2 = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])

test_inputs_3 = np.array([[2,0],[0,2],[-2,0],[0,-2],[0.5,0.5],[-0.5,0.5],[0.5,-0.5],[-0.5,-0.5]])
test_out_3 = np.array([[1],[1],[1],[1],[0],[0],[0],[0]])

# function to use the trained weights to predict the testing output
def predict(testing_array, target_test):
    
    # evaluating the hidden layer seet for the test set
    h_l= np.dot(testing_array,hidden_weights)
    temp = sigmoid(h_l+hidden_bias)

    # evaluating the output for the test set
    out = np.dot(temp,output_weights)
    out_final = sigmoid(out+output_bias)
    print("\nOutput from nn from test set using the trained set:\n ",out_final)

# using the predict function to test the the new input array with new unseen set
predict(test_inputs_1, test_out_1)
predict(test_inputs_2, test_out_2)
predict(test_inputs_3, test_out_3)

# after testing on three test sets it is clear that with 4 units in hidden layer the systme learns the pattern exactly and is able to generalie on different sets of data
# taking 4 units i the hidden layer the system is able to draw the right decision boundaries needed to seperate the classes present on the axis from the points away from the axis.
