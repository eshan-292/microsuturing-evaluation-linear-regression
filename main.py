import getopt, sys  # For commande line arguments


import pandas as pd
import numpy as np




# Function to read the data from the csv file
def read_data(file):
    output = []
    input = []
    with open(file, 'r') as f:
        for line in f:
            # print(line)
            line = line.split(',')
            line[2049] = line[2049].replace('\n', '')   # removing \n
            output.append(line[1])
            input.append(line[2:2050])
            
    # print(input, output)
    return input, output


# Function to read the data from the csv file
def read_data_output(file):
    output = []
    input = []
    samplenames = []
    with open(file, 'r') as f:
        for line in f:
            # print(line)
            line = line.split(',')
            line[2048] = line[2048].replace('\n', '')   # removing \n
            samplenames.append(line[0])
            # output.append(line[0])
            input.append(line[1:2049])
            
    # print(input, output)
    return input, samplenames




def normalise(x):
    x = (x - np.mean(x, axis=0)) / (np.std(x, axis=0))
    return x

    






# #LINEAR REGRESSION

def linear_regression_grad_descent(train_input, train_output, learning_rate, max_it, reltol, validation_input, validation_output):
    
    


    # Preprocessing the data 
    train_input = np.array(train_input, dtype=float)
    train_output = np.array(train_output, dtype=float).reshape(train_input.shape[0], 1)     # Converting the output to a column vector
    validation_input = np.array(validation_input, dtype=float)
    validation_output = np.array(validation_output, dtype=float).reshape(validation_input.shape[0], 1)     # Converting the output to a column vector
    
    # Normalising the data
    # train_input = normalise(train_input)
    
    # train_output = normalise(train_output)
    


    
    train_input = np.concatenate((np.ones((train_input.shape[0], 1), dtype=float), train_input), axis=1)     # Appending 1 to each data element to account for the bias
    validation_input = np.concatenate((np.ones((validation_input.shape[0], 1), dtype=float), validation_input), axis=1)     # Appending 1 to each data element to account for the bias
    

    # Initializing the weights
    weights = np.array([0] * train_input.shape[1], dtype=float).reshape(train_input.shape[1], 1)
    
    mse = 0    # Initializing the mean squared error
    mse_val = 0
    
    training_loss_vector = []
    validation_loss_vector = []


    # Calculating weights iteratively
    for i in range(max_it):
        y = np.dot(train_input, weights)
        y_val = np.dot(validation_input, weights)
        error = y - train_output        # Calculating the difference in predicted and output values
        error_val = y_val - validation_output
        # print(error)
        mse_prev = mse                  # Updating the mse of previous iteration
        mse = np.sum(error**2)/ train_input.shape[0]      # Calculating the mean squared error of current iteration
        mse_val_prev = mse_val
        mse_val = np.sum(error_val**2)/ validation_input.shape[0]

        if i%(max_it/20) == 0:
            training_loss_vector.append(str(mse))
            validation_loss_vector.append(str(mse_val))


        # print(mse)
        mae = np.sum(abs(error))/train_input.shape[0]      # Calculating the mean absolute error of current iteration
        
        # Normalised gradient
        # grad = normalise(np.dot(train_input.T, error)/train_input.shape[0] )
        
        # Unnormalised gradient
        grad = np.dot(train_input.T, error)/train_input.shape[0]
        
        # print(grad.shape)
        weights = weights - learning_rate * grad

        if i > 0 and (abs(mse_val - mse_val_prev))/mse_val_prev < reltol:
            break

    
    # print("Linear Regression Training MAE: ", mae)
    # print("Linear Regression Training MSE: ", mse)

    
    return weights, training_loss_vector, validation_loss_vector

# Predicting the output for given input data using the learned weights
def linear_regression_predict(input, weights):
    # Preprocessing the input data
    input = np.array(input, dtype=float)
    input = np.concatenate((np.ones((len(input), 1), dtype=float), input), axis=1) 
    
    # Calculating the output
    output = np.dot(input, weights)
    return output



def test_linear_regression(linear_regression_model_weights,filename):

    # Generating the validation input and output
    validation_input, validation_output = read_data(filename)
    validation_output = np.array(validation_output, dtype=float).reshape(len(validation_output), 1)

    output = linear_regression_predict(validation_input, linear_regression_model_weights)

    # Calculating the mse and mae
    mae = np.mean(abs(output - validation_output))
    mse = np.sum((output - validation_output)**2)/len(output)

    # print("Linear Regression Validation MAE: ", mae)
    # print("Linear Regression Validation MSE: ", mse)





# Function to read the data from the csv file
def read_ndata(file):
    output = []
    input = []
    with open(file, 'r') as f:
        for line in f:
            
                
            line = line.split(',')
            line[-1] = line[-1].replace('\n', '')   # removing \n
            output.append(line[-1])
            input.append(line[0:-1])
            
    # print(input, output)
    return input, output






# it = 1000

# # Generating the training input and output
# train_input, train_output = read_ndata("Generalization/100_d_train.csv")
# validation_input, validation_output = read_ndata("Generalization/100_d_test.csv")

# # print(train_input)
# # print(train_output)

# train_input = np.array(train_input, dtype=float)
# train_output = np.array(train_output, dtype=float).reshape(len(train_output), 1)




# # # Training the linear regression using gradient descent model
# linear_regression_model_weights, training_loss_vector, validation_loss_vector = linear_regression_grad_descent(train_input, train_output, 0.001, it, 0.0000000000001, validation_input, validation_output)
# # # print("Training MSE", training_loss_vector)
# # # print("Validation MSE", validation_loss_vector)

# validation_input = np.array(validation_input, dtype=float)
# validation_output = np.array(validation_output, dtype=float).reshape(len(validation_output), 1)
# output = linear_regression_predict(validation_input, linear_regression_model_weights)
# mse_val = np.sum((output - validation_output)**2)/len(output)
# print("Linear Regression Validation MSE: ", mse_val)


# print(train_input.shape)
# print(train_output.shape)





# # Splitting the dataset into one fourths for visualisation part

# from sklearn.model_selection import train_test_split

# train_input, train_output = read_data("train.csv")
# train_input, train_output = np.array(train_input, dtype=float), np.array(train_output, dtype=float)

# frac = 0.5     # Fraction of data to be used for training
# train_input_1, train_input_2, train_output_1, train_output_2 = train_test_split(train_input, train_output, train_size=frac, random_state=1)



# linear_regression_model_weights_1, training_loss_vector, validation_loss_vector = linear_regression_grad_descent(train_input_1, train_output_1, 0.001, it, 0.0000000000001, validation_input, validation_output)
# linear_regression_model_weights_2, training_loss_vector, validation_loss_vector = linear_regression_grad_descent(train_input_2, train_output_2, 0.001, it, 0.0000000000001, validation_input, validation_output)

# output_1 = linear_regression_predict(train_input_1, linear_regression_model_weights_1)
# output_2 = linear_regression_predict(train_input_2, linear_regression_model_weights_2)

# print("Mean Absolute Difference= ", np.mean(abs(output_1 - output_2)))





# # Training the linear regression using gradient descent model
# linear_regression_model_weights, training_loss_vector, validation_loss_vector = linear_regression_grad_descent(train_input, train_output, 0.001, it, 0.0000000000001, validation_input, validation_output)
# # print("Training MSE", training_loss_vector)
# # print("Validation MSE", validation_loss_vector)

# output = linear_regression_predict(train_input, linear_regression_model_weights)

# train_output = np.array(train_output, dtype=float).reshape(len(train_output), 1)

# # Calculating the MSE Loss per score
# total_mse_loss = [0]*10
# avg_mse_loss = [0]*10

# for i in range(1,10):
#     # print("MSE Loss for score ", i, " is ", np.sum((output - train_output)**2)/len(output))
#     total_mse_loss[i] = 0
#     cnt = 0
#     for j in range(len(output)):
#         if train_output[j] == i:
#             total_mse_loss[i] += float((output[j] - train_output[j])**2)
#             cnt+=1
#     avg_mse_loss[i] = total_mse_loss[i]/cnt
#             # print("MSE Loss for score ", i, " is ", (output[j] - train_output[j])**2)
#             # break

# print(total_mse_loss)
# print(avg_mse_loss)


# # Testing the linear regression model using validation data
# test_linear_regression(linear_regression_model_weights, "validation.csv")



# # Writing to csv file to plot data

# import csv 
# # name of csv file 
# filename = "plot.csv"
    
# # writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile, delimiter=',') 
        
#     csvwriter.writerow(["A", "B", "C"])


#     # i=0
#     # for j in range(len(training_loss_vector)):
#     #     # print(element)
#     #     csvwriter.writerow([int(i), training_loss_vector[j], validation_loss_vector[j]])
#     #     i+= it/20


#     for i in range(1,10):
#         csvwriter.writerow([i, total_mse_loss[i] ,  avg_mse_loss[i]])













#--------------------------------------------------------------------------------------------------------------------------------------









# RIDGE REGRESSION

def ridge_regression_grad_descent(train_input, train_output, learning_rate, max_it, reltol,lambda_par, validation_input, validation_output):
    
    # Preprocessing the data 
    train_input = np.array(train_input, dtype=float)
    train_output = np.array(train_output, dtype=float).reshape(train_input.shape[0], 1)     # Converting the output to a column vector
    validation_input = np.array(validation_input, dtype=float)
    validation_output = np.array(validation_output, dtype=float).reshape(validation_input.shape[0], 1)     # Converting the output to a column vector
    
    # Normalising the data
    # train_input = normalise(train_input)
    
    # train_output = normalise(train_output)
    


    
    train_input = np.concatenate((np.ones((train_input.shape[0], 1), dtype=float), train_input), axis=1)     # Appending 1 to each data element to account for the bias
    validation_input = np.concatenate((np.ones((validation_input.shape[0], 1), dtype=float), validation_input), axis=1)     # Appending 1 to each data element to account for the bias
    

    # Initializing the weights
    weights = np.array([0] * train_input.shape[1], dtype=float).reshape(train_input.shape[1], 1)
    
    mse = 0    # Initializing the mean squared error
    mse_val = 0
    
    training_loss_vector = []
    validation_loss_vector = []

    
    # Calculating weights iteratively
    for i in range(max_it):
        y = np.dot(train_input, weights)
        y_val = np.dot(validation_input, weights)
        error = y - train_output        # Calculating the difference in predicted and output values
        error_val = y_val - validation_output
        # print(error)
        mse_prev = mse                  # Updating the mse of previous iteration
        mse = np.sum(error**2)/ train_input.shape[0]      # Calculating the mean squared error of current iteration
        mse_val_prev = mse_val
        mse_val = np.sum(error_val**2)/ validation_input.shape[0]

        if i%(max_it/20) == 0:
            training_loss_vector.append(str(mse))
            validation_loss_vector.append(str(mse_val))


        # print(mse)
        mae = np.sum(abs(error))/train_input.shape[0]      # Calculating the mean absolute error of current iteration
        

        
        #Unnormalised Gradient
        grad = (np.dot(train_input.T, error)/train_input.shape[0] ) + (lambda_par * weights/train_input.shape[0]) 
        
        # print(grad.shape)
        weights = weights - learning_rate * grad

        if i > 0 and (abs(mse_val - mse_val_prev))/mse_val_prev < reltol:
            break



    # print("Ridge Regression Training MAE: ", mae)
    # print("Ridge Regression Training MSE: ", mse)

    # print("Ridge Regression Validation MSE: ", mse_val)

    return weights, training_loss_vector, validation_loss_vector


# Predicting the output for given input data using the learned weights
def ridge_regression_predict(input, weights):
    # Preprocessing the input data
    input = np.array(input, dtype=float)
    input = np.concatenate((np.ones((len(input), 1), dtype=float), input), axis=1) 
    
    # Calculating the output
    output = np.dot(input, weights)
    return output



def test_ridge_regression(ridge_regression_model_weights, filename):
    # Generating the validation input and output
    validation_input, validation_output = read_data(filename)
    validation_output = np.array(validation_output, dtype=float).reshape(len(validation_output), 1)

    output = ridge_regression_predict(validation_input, ridge_regression_model_weights)



    # Calculating the mse and mae
    mae = np.mean(abs(output - validation_output))
    mse = np.sum((output - validation_output)**2)/len(output)

    # print("Ridge Regression Validation MAE: ", mae)
    # print("Ridge Regression Validation MSE: ", mse)



    
# it = 100
# # Generating the training input and output
# train_input, train_output = read_data("train.csv")
# validation_input, validation_output = read_data("validation.csv")



# # Splitting the dataset into one fourths for visualisation part

# from sklearn.model_selection import train_test_split

# train_input, train_output = read_data("train.csv")
# train_input, train_output = np.array(train_input, dtype=float), np.array(train_output, dtype=float)

# frac = 0.5     # Fraction of data to be used for training
# train_input_1, train_input_2, train_output_1, train_output_2 = train_test_split(train_input, train_output, train_size=frac, random_state=1)



# ridge_regression_model_weights_1, training_loss_vector, validation_loss_vector = ridge_regression_grad_descent(train_input_1, train_output_1, 0.001, it, 0.0000000000001,25, validation_input, validation_output)
# ridge_regression_model_weights_2, training_loss_vector, validation_loss_vector = ridge_regression_grad_descent(train_input_2, train_output_2, 0.001, it, 0.0000000000001,25,  validation_input, validation_output)

# output_1 = ridge_regression_predict(train_input_1, ridge_regression_model_weights_1)
# output_2 = ridge_regression_predict(train_input_2, ridge_regression_model_weights_2)

# print("Mean Absolute Difference= ", np.mean(abs(output_1 - output_2)))






# # Training the ridge regression using gradient descent model
# ridge_regression_model_weights, training_loss_vector, validation_loss_vector = ridge_regression_grad_descent(train_input, train_output, 0.001, it, 0.00000000001, 25, validation_input, validation_output)


# # Testing the ridge regression model using validation data
# test_ridge_regression(ridge_regression_model_weights, "validation.csv")




# # Writing to csv file to plot data

# import csv 
# # name of csv file 
# filename = "plot.csv"
    
# # writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile, delimiter=',') 
        
#     csvwriter.writerow(["A", "B"])


#     i=0
#     for j in range(len(training_loss_vector)):
#         # print(element)
#         csvwriter.writerow([int(i), training_loss_vector[j], validation_loss_vector[j]])
#         i+= it/20












#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------



# # USING SKLEARN




# #LINEAR REGRESSION
# from sklearn.linear_model import LinearRegression

# #Generating the training input and output
# train_input, train_output = read_data("train.csv")
# train_input = np.array(train_input, dtype=float)
# train_output = np.array(train_output, dtype=float)     # Converting the output to a column vector

# sklearn_lin_reg = LinearRegression().fit(train_input, train_output)
# # sklearn_lin_reg.score(train_input, train_output)

# #Generating the validation input and output
# validation_input, validation_output = read_data("validation.csv")
# validation_input = np.array(validation_input, dtype=float)
# validation_output = np.array(validation_output, dtype=float)     # Converting the output to a column vector
# output = sklearn_lin_reg.predict(validation_input)
# # print(output)

# #Calculating the mse and mae
# mae = np.mean(abs(output - validation_output))
# mse = np.mean((output - validation_output)**2)

# print("Scikitlearn Linear Regression MAE: ", mae)
# print("Scikitlearn Linear Regression MSE: ", mse)




# # RIDGE REGRESSION
# from sklearn.linear_model import Ridge

# #Generating the training input and output
# train_input, train_output = read_data("train.csv")
# train_input = np.array(train_input, dtype=float)
# train_output = np.array(train_output, dtype=float)     # Converting the output to a column vector

# sklearn_ridge_reg = Ridge(alpha=5).fit(train_input, train_output)

# # Generating the validation input and output
# validation_input, validation_output = read_data("validation.csv")
# validation_input = np.array(validation_input, dtype=float)
# validation_output = np.array(validation_output, dtype=float)     # Converting the output to a column vector
# output = sklearn_ridge_reg.predict(validation_input)
# # print(output)

# #Calculating the mse and mae
# mae = np.mean(abs(output - validation_output))
# mse = np.mean((output - validation_output)**2)

# print("Scikitlearn Ridge Regression MAE: ", mae)
# print("Scikitlearn Ridge Regression MSE: ", mse)










#--------------------------------------------------------------------------------------------------------------------------------------





# # FEATURE SELECTION USING SKLEARN


# from sklearn.feature_selection import SelectKBest
# from sklearn.feature_selection import f_classif



# # Training the models on the selected features
# it = 5000


# # SelectKBest
# # Generating the training input and output
# train_input, train_output = read_data("train.csv")
# train_input = np.array(train_input, dtype=float)
# train_output = np.array(train_output, dtype=float)     # Converting the output to a column vector
# validation_input, validation_output = read_data("validation.csv")
# validation_input = np.array(validation_input, dtype=float)
# validation_output = np.array(validation_output, dtype=float)     # Converting the output to a column vector

# # print(train_input.shape)
# # print(train_output.shape)

# train_input_10 = SelectKBest(f_classif, k=10).fit_transform(train_input, train_output)       #Selecting the 10 best features
# validation_input_10 = SelectKBest(f_classif, k=10).fit_transform(validation_input, validation_output)       #Selecting the 10 best features

# # print(train_input_10.shape)


# # LINEAR REGRESSION
# # Training the linear regression using gradient descent model

# linear_regression_model_weights, training_loss_vector, validation_loss_vector  = linear_regression_grad_descent(train_input_10, train_output, 0.001, it, 0.0000001, validation_input_10, validation_output)

# # Writing to csv file to plot data

# import csv 
# # name of csv file 
# filename = "plot.csv"
    
# # writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile, delimiter=',') 
        
#     csvwriter.writerow(["A", "B"])


#     i=0
#     for j in range(len(training_loss_vector)):
#         # print(element)
#         csvwriter.writerow([int(i), training_loss_vector[j], validation_loss_vector[j]])
#         i+= it/20













# # SelectFromModel
# from sklearn.feature_selection import SelectFromModel
# from sklearn.linear_model import Ridge






# it = 1000
# # Generating the training input and output
# train_input, train_output = read_data("train.csv")
# validation_input, validation_output = read_data("validation.csv")

# train_input = np.array(train_input, dtype=float)
# train_output = np.array(train_output, dtype=float)     # Converting the output to a column vector
# validation_input = np.array(validation_input, dtype=float)
# validation_output = np.array(validation_output, dtype=float)     # Converting the output to a column vector

# selector =  SelectFromModel(estimator=Ridge(), max_features=2048).fit(train_input, train_output)
# train_input_10 = selector.transform(train_input)
# # selector = SelectFromModel(estimator=Ridge(), max_features=10).fit(validation_input, validation_output)
# validation_input_10 = selector.transform(validation_input)

# # Training the ridge regression using gradient descent model
# ridge_regression_model_weights, training_loss_vector, validation_loss_vector = ridge_regression_grad_descent(train_input_10, train_output, 0.001, it, 0.00000000001, 25, validation_input_10, validation_output)






# # Writing to csv file to plot data

# import csv 
# # name of csv file 
# filename = "plot.csv"
    
# # writing to csv file 
# with open(filename, 'w') as csvfile: 
#     # creating a csv writer object 
#     csvwriter = csv.writer(csvfile, delimiter=',') 
        
#     csvwriter.writerow(["A", "B"])


#     i=0
#     for j in range(len(training_loss_vector)):
#         # print(element)
#         csvwriter.writerow([int(i), training_loss_vector[j], validation_loss_vector[j]])
#         i+= it/20

















#--------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------

















# LOGISTIC REGRESSSION FOR CLASSIFICATION


def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression_grad_descent(train_input, train_output, num_classes, learning_rate, max_it, reltol):
    
    # Preprocessing the data 
    train_input = np.array(train_input, dtype=float)
    train_output = np.array(train_output, dtype=int)    
    
    # # Converting train_output into one-hot encoded representation
    # train_output_encoded = np.zeros((train_output.shape[0], num_classes))
    # for i, c in enumerate(np.unique(train_output)):
    #     train_output_encoded[:, i] = (train_output == c).astype(int)
    
    
    
    train_input = np.concatenate((np.ones((train_input.shape[0], 1), dtype=float), train_input), axis=1)     # Appending 1 to each data element to account for the bias
    



    # Initializing the weights
    
    # weights = np.zeros((train_input.shape[1], num_classes))
    weights =  np.zeros((num_classes, train_input.shape[1]))

    cost = 0   



    for c in range(1,num_classes+1):
        # train_output_c = np.array([1 if y_i == c else 0 for y_i in train_output])
        train_output_c = np.where(train_output == c, 1, 0)
        weight = np.zeros(train_input.shape[1])
        for i in range(max_it):
            y = sigmoid(np.dot(train_input, weight))
            error = y - train_output_c
            grad = np.dot(train_input.T, error)/train_input.shape[0]
            weight = weight - learning_rate * grad
        
        weights[c-1, :] = weight
    
    
    

    # # Calculating weights iteratively
    # for i in range(max_it):
    #     weights[:, -1] = 0 
    #     y = sigmoid(np.dot(train_input, weights))       # Predicted probabilites
    #     error = y - train_output        # Calculating the difference in predicted and output values
        
    #     # print(error)

    #     cost_prev = cost
    #     cost = -np.sum(train_output_encoded * np.log(y) + (1 - train_output_encoded) * np.log(1 - y))/train_input.shape[0]
    #     grad = np.dot(train_input.T, error)/train_input.shape[0] 
    #     # print(grad.shape)
    #     weights = weights - learning_rate * grad

    #     if i > 0 and abs(cost - cost_prev) < reltol:
    #         break

    
    # print("Training Logistic Regression Error: ", cost)

    
    return weights

# Predicting the output for given input data using the learned weights
def logistic_regression_predict(input, weights):
    # Preprocessing the input data
    input = np.array(input, dtype=float)
    input = np.concatenate((np.ones((len(input), 1), dtype=float), input), axis=1) 
    
    # Calculating the output
    output = sigmoid(np.dot(input, weights.T))

    # print(output)
    
    # Calculating the probabilities of the last class
    np.concatenate((output, np.zeros((output.shape[0], 1), dtype=float)), axis=1) 
    for i in range(output.shape[0]):
        output[i][-1] = 1-(np.sum(output[i]))    

    # Calculating the class with maximum probability
    output = np.argmax(output, axis=1)
    for i in range(len(output)):
        output[i] += 1

    return output






# # Generating the training input and output
# train_input, train_output = read_data("train.csv")

# # Training the logistic regression using gradient descent model
# logistic_regression_model_weights = logistic_regression_grad_descent(train_input, train_output, 8, 0.001, 1000, 0.001)


# # Generating the validation input and output
# validation_input, validation_output = read_data("validation.csv")
# validation_output = np.array(validation_output, dtype=int)
# print(validation_output)

# output = logistic_regression_predict(validation_input, logistic_regression_model_weights)

# print(output)

# error = 0
# for i in range(len(output)):
#     if output[i] != validation_output[i]:
#         error += 1

# print("Error: ", error/len(output))


# mse = np.sum((output - validation_output)**2)/len(output)
# print("MSE: ", mse)
















# Handling command line arguments
argumentList = sys.argv[1:]

options = "hmo:"


# Long options
long_options = ["train_path=", "val_path=", "test_path=", "out_path=", "section="]
section = 0

try:
    # Parsing argument
    arguments, values = getopt.getopt(argumentList, options, long_options)
     
    # checking each argument
    for currentArgument, currentValue in arguments:
        # print (currentArgument)
        # print(currentValue)

        if currentArgument in ("--train_path"):
            path_to_train = currentValue
             
        elif currentArgument in ("--val_path"):
            path_to_val = currentValue
             
        elif currentArgument in ("--test_path"):
            path_to_test = currentValue
        
        elif currentArgument in ("--out_path"):
            path_to_out = currentValue
        
        elif currentArgument in ("--section"):
            section = currentValue
             
except getopt.error as err:
    # output error
    print (str(err))


# path_to_train = "train.csv"

# path_to_test = "test.csv"
# path_to_val = "validation.csv"
# path_to_out = "out.csv"

# section=5

# print(section)

if section == '1':
    #Linear Regression

    it = 1000

    # Generating the training input and output
    train_input, train_output = read_data(path_to_train)
    validation_input, validation_output = read_data(path_to_val)
    test_input, samplenames = read_data_output(path_to_test)

    test_input = np.array(test_input, dtype=float)
    # test_output = np.array(test_output, dtype=float).reshape(len(test_output), 1)
    

    # print(train_input)
    # print(train_output)

    # train_input = np.array(train_input, dtype=float)
    # train_output = np.array(train_output, dtype=float).reshape(len(train_output), 1)




    # # Training the linear regression using gradient descent model
    linear_regression_model_weights, training_loss_vector, validation_loss_vector = linear_regression_grad_descent(train_input, train_output, 0.001, it, 0.00000000001, validation_input, validation_output)


    # Generating results on Test data
    output = linear_regression_predict(test_input, linear_regression_model_weights)
    # mse_val = np.sum((output - validation_output)**2)/len(output)
    # print("Linear Regression Validation MSE: ", mse_val)


    # Writing to the output csv file 

    import csv 
    # name of csv file 
    filename = path_to_out
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter=',') 
            
        # csvwriter.writerow(["A", "B", "C"])

        

        for i in range(len(samplenames)):
            row = []
            row.append(samplenames[i])
            row.append(float(output[i])) 
            csvwriter.writerow(row)


elif section == '2' :      
    
    it = 1000

    # Generating the training input and output
    train_input, train_output = read_data(path_to_train)
    validation_input, validation_output = read_data(path_to_val)
    test_input, samplenames = read_data_output(path_to_test)

    test_input = np.array(test_input, dtype=float)



    # Training the ridge regression using gradient descent model
    ridge_regression_model_weights, training_loss_vector, validation_loss_vector = ridge_regression_grad_descent(train_input, train_output, 0.001, it, 0.00000000001, 25, validation_input, validation_output)


    output = ridge_regression_predict(test_input, ridge_regression_model_weights)





    # Writing to the output csv file 

    import csv 
    # name of csv file 
    filename = path_to_out
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter=',') 
            
        # csvwriter.writerow(["A", "B", "C"])

        

        for i in range(len(samplenames)):
            row = []
            row.append(samplenames[i])
            row.append(float(output[i])) 
            csvwriter.writerow(row)



    

elif section=='5':

    # Generating the training input and output
    train_input, train_output = read_data(path_to_train)
    # validation_input, validation_output = read_data(path_to_val)
    test_input, samplenames = read_data_output(path_to_test)

    test_input = np.array(test_input, dtype=float)


    # Training the logistic regression using gradient descent model
    logistic_regression_model_weights = logistic_regression_grad_descent(train_input, train_output, 8, 0.001, 1000, 0.001)


    # Testing
    output = logistic_regression_predict(test_input, logistic_regression_model_weights)

    # print(output)



    # Writing to the output csv file 

    import csv 
    # name of csv file 
    filename = path_to_out
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter=',') 
            
        # csvwriter.writerow(["A", "B", "C"])

        

        for i in range(len(samplenames)):
            row = []
            row.append(samplenames[i])
            row.append(int(output[i])) 
            csvwriter.writerow(row)



