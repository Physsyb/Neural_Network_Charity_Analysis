# Neural_Network_Charity_Analysis
# Project Overview
Bek’s come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

![header](https://user-images.githubusercontent.com/76136277/117586023-a6f3ea80-b0e3-11eb-9e63-27c7493ed5fc.png)

This new assignment consists of three technical analysis deliverables and a written report. You will submit the following:
* Deliverable 1: Preprocessing Data for a Neural Network Model
* Deliverable 2: Compile, Train, and Evaluate the Model
* Deliverable 3: Optimize the Model

# Resources
* Data source: `charity_data.csv`
* Data Tools/Software: Python 3.8.5, Jupyter Notebook 6.1.4 and Pandas

## Deliverable 1: Preprocessing Data for a Neural Network Model
> Using your knowledge of Pandas and the Scikit-Learn’s `StandardScaler()`, you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

### Requirements
1. The `EIN` and `NAME` columns have been dropped
![1](https://user-images.githubusercontent.com/76136277/117586344-a0ff0900-b0e5-11eb-85f4-3a00a478ad64.PNG)

2. The columns with more than 10 unique values have been grouped together 
![2](https://user-images.githubusercontent.com/76136277/117586349-a52b2680-b0e5-11eb-9925-eea38aa9ac7a.PNG)

3.  The categorical variables have been encoded using one-hot encoding 
![3](https://user-images.githubusercontent.com/76136277/117586354-aa887100-b0e5-11eb-9286-80a306cd2c47.PNG)

4. The preprocessed data is split into features and target arrays 
5. The preprocessed data is split into training and testing datasets 
![4](https://user-images.githubusercontent.com/76136277/117586358-aeb48e80-b0e5-11eb-8a94-a2d50eaba923.PNG)

6. The numerical values have been standardized using the `StandardScaler()` module 
![5](https://user-images.githubusercontent.com/76136277/117586361-b2e0ac00-b0e5-11eb-8b0e-1670b095127e.PNG)

## Deliverable 2: Compile, Train, and Evaluate the Model
> Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

### Requirements
The neural network model using Tensorflow Keras contains working code that performs the following steps:
1. The number of layers, the number of neurons per layer, and activation function are defined 
2. An output layer with an activation function is created 
3. There is an output for the structure of the model 
4. There is an output of the model’s loss and accuracy 
![6](https://user-images.githubusercontent.com/76136277/117586532-ba548500-b0e6-11eb-9e4f-356459af3199.PNG)

5. The model's weights are saved every 5 epochs 
![7](https://user-images.githubusercontent.com/76136277/117586547-d0fadc00-b0e6-11eb-8463-eb37893f90ac.PNG)

6. The results are saved to an HDF5 file 
![8](https://user-images.githubusercontent.com/76136277/117586555-d5bf9000-b0e6-11eb-9cce-5cf80853fdc6.PNG)

## Optimize the Model
> Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

### Requirements
The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
1. Noisy variables are removed from features
2. Additional neurons are added to hidden layers 
3. Additional hidden layers are added 
4. The activation function of hidden layers or output layers is changed for optimization 
      * #### Attempt 1
          ![9](https://user-images.githubusercontent.com/76136277/117586812-88dcb900-b0e8-11eb-9db8-49bb080ef13d.PNG)

      * #### Attempt 2
          ![12](https://user-images.githubusercontent.com/76136277/117586864-e7099c00-b0e8-11eb-87a8-7b67695ade07.PNG)

      * #### Attempt 3
          ![13](https://user-images.githubusercontent.com/76136277/117586874-ef61d700-b0e8-11eb-974c-7d9a17fd5e1e.PNG)

5. The model's weights are saved every 5 epochs 
![10](https://user-images.githubusercontent.com/76136277/117586816-8e3a0380-b0e8-11eb-8270-05baf275f8f2.PNG)

6. The results are saved to an HDF5 file 
![11](https://user-images.githubusercontent.com/76136277/117586820-91cd8a80-b0e8-11eb-9bc8-886f7ffafcf5.PNG)

# Results
1. Data Preprocessing
    * What variable(s) are considered the target(s) for your model?
       * IS_SUCCESSFUL 
       
    * What variable(s) are considered to be the features for your model?
       * US
       * INCOME_AMT
       * SPECIAL_CONSIDERATIONS
       * ASK_AMT
       * APPLICATION_TYPE
       * AFFILIATION
       * CLASSIFICATION
       * USE_CASE

    * What variable(s) are neither targets nor features, and should be removed from the input data?
       * NAME
       * EIN
      
2. Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
       * I have 2 hidden layers fro my neural network. The first layer has 80 neurons while the second has 30 layers. There is also an output layer. The first and second hidden layer have the "relu" activation function and the activation function for the output layer is "sigmoid."
![11](https://user-images.githubusercontent.com/76136277/117587124-6055be80-b0ea-11eb-8692-fe2b1464f2cd.PNG)

    * Were you able to achieve the target model performance?
      * The model was not able to reach the target 75%. The accuracy for my model was 52%.
            ![0](https://user-images.githubusercontent.com/76136277/117587375-a5c6bb80-b0eb-11eb-8000-85801e2070ee.PNG)

    * What steps did you take to try and increase model performance?
      * Additional neurons to hidden layers
      * Increasing the number of hidden layers to include a 3rd layer
          ![12](https://user-images.githubusercontent.com/76136277/117587506-4e751b00-b0ec-11eb-8703-28e3c7ce62a3.PNG)
          ![000](https://user-images.githubusercontent.com/76136277/117587516-5634bf80-b0ec-11eb-8aba-bd2980828889.PNG)
      
      * Changing the activation functions: tried linear, tanh, sigmoid for a combination of hidden layers and output layer

# Summary
The model ended up with the accuracy score of about 50% after optimization. The initial neural network had a accuracy score of 52%. This loss in accuracy can be explained from the fact that the model was over fitted. 
