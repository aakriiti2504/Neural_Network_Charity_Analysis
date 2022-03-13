# Neural_Network_Charity_Analysis


## Background
Bex is a data scientist and programmer for the non-profit foundation, 'Alphabet Soup'. They are full enthropic foundation dedicated to helping organizations that protect the environment, improve people's well being and unify the world. Alphabet Soup has raised and donated over 10 billion dollars in the past 20 years. This money has been used to invest in life saving technologies and organize reforce station groups around the world . bex is responsible for data collection and anlaysis for the entire organization. Her job is to analyze the impact of each donation and vet potential recipients. This helps ensure that the foundation money is being used effectively. Unfortunately, not every donation that the company makes is impactful. In some cases an organization will take the money and disappear. As a result, the president of 'Alphabet Soups' Andy Glad, has asked Bex to predict which organizations are worth donating to and which ones are high risk. He wants her to create a mathematical data driven solution that can do this accurately. Bex has this idea that this problem is too complex for the statistical and machine learning models that she has used. Instead she will design and train a deep learning neural network. This model will evaluate all types of input data and produce a clear decision making result. I am going to help Bex learn about Neural Networks and how to design and train these models using the python TensorFlow Libraries. We will then test and optimize the models using statistics and and machine learning. We will create a robust and deep learning neural network capable of interpreting large and complex datasets. This will help Bex and Alphabet Soup decide which organizations should receive donations. 

![1](https://user-images.githubusercontent.com/23488019/157788996-5130eb23-b24b-44c1-9bd8-92ec1db19632.PNG)
![2](https://user-images.githubusercontent.com/23488019/157789004-8726abb2-68aa-430b-b57f-9c2b21426fc9.PNG)


## Neural Network:
Neural networks (also known as artificial neural networks, or ANN) are a set of algorithms that are modeled after the human brain. They are an advanced form of machine learning that recognizes patterns and features in input data and provides a clear quantitative output. In its simplest form, a neural network contains layers of neurons, which perform individual computations. These computations are connected and weighed against one another until the neurons reach the final layer, which returns a numerical result, or an encoded categorical result.

![data-19-1-1-1-deep-neural-network-includes-input-and-output-layer](https://user-images.githubusercontent.com/23488019/157789081-760491cd-75fc-4b77-a3ae-af283fd50bd7.gif)
Neural networks are particularly useful in data science because they serve multiple purposes. One way to use a neural network model is to create a classification algorithm that determines if an input belongs in one category versus another. Alternatively neural network models can behave like a regression model, where a dependent output variable can be predicted from independent input variables. Therefore, neural network models can be an alternative to many of the models we have learned throughout the course, such as random forest, logistic regression, or multiple linear regression.

There are a number of advantages to using a neural network instead of a traditional statistical or machine learning model. For instance, neural networks are effective at detecting complex, nonlinear relationships. Additionally, neural networks have greater tolerance for messy data and can learn to ignore noisy characteristics in data. The two biggest disadvantages to using a neural network model are that the layers of neurons are often too complex to dissect and understand (creating a black box problem), and neural networks are prone to overfitting (characterizing the training data so well that it does not generalize to test data effectively). However, both of the disadvantages can be mitigated and accounted for. Neural networks have many practical uses across multiple industries. In the finance industry, neural networks are used to detect fraud as well as trends in the stock market. Retailers like Amazon and Apple are using neural networks to classify their consumers to provide targeted marketing as well as behavior training for robotic control systems. Due to the ease of implementation, neural networks also can be used by small businesses and for personal use to make more cost-effective decisions on investing and purchasing business materials. Neural networks are scalable and effective.

## Overview
Beks has come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

## Purpose - What I am Creating:
This project consists of the following Deliverables :
1. Deliverable 1: Preprocessing Data for a Neural Network Model
2. Deliverable 2: Compile, Train, and Evaluate the Model
3. Deliverable 3: Optimize the Model
4. Deliverable 4: A Written Report on the Neural Network Model (README.md)


### 1. Deliverable 1: Preprocessing Data for a Neural Network Model
1. Read in the charity_data.csv to a Pandas DataFrame
 
![11](https://user-images.githubusercontent.com/23488019/158046023-caafa1f7-9efb-4f98-8bfe-04678ab0784c.PNG)

2. Dropping the EIN and NAME columns since both these columns do not affect the success or the failure rate. 
3. Determine the number of unique values for each column.

![12](https://user-images.githubusercontent.com/23488019/158046030-19cc2e1c-24ce-49f5-b5d0-4c602bbf61ac.PNG)

4. For those columns that have more than 10 unique values, determine the number of data points for each unique value.

![13](https://user-images.githubusercontent.com/23488019/158046060-d985a2d5-2394-4d92-abed-e98dff20b449.PNG)

5. Create a density plot to determine the distribution of the column values.
6. Use the density plot to create a cutoff point to bin "rare" categorical variables together in a new column, Other, and then check if the binning was successful.
![14](https://user-images.githubusercontent.com/23488019/158046070-61322244-b4cb-46db-9c80-45184eda953a.PNG)
![15](https://user-images.githubusercontent.com/23488019/158046074-f32b06e0-1291-414e-bb1d-7105c3be5b0e.PNG)
![16](https://user-images.githubusercontent.com/23488019/158046078-61ef1a0e-c58f-4ff9-88c8-59e1652f91ca.PNG)

7. Generate a list of categorical variables.
8. Encode categorical variables using one-hot encoding, and place the variables in a new DataFrame.
9. Merge the one-hot encoding DataFrame with the original DataFrame, and drop the originals.
![17](https://user-images.githubusercontent.com/23488019/158046082-894a4fc6-35cb-480b-84ef-ff44c2934f6a.PNG)

10. Split the preprocessed data into features and target arrays.
11. Split the preprocessed data into training and testing datasets.
![18](https://user-images.githubusercontent.com/23488019/158046086-16df65c5-5466-4927-adec-7eeabb43a746.PNG)

12. Standardize numerical variables using Scikit-Learn’s StandardScaler class, then scale the data.



### 2. Delverable 2: Compile, Train, and Evaluate the Model
Using TensorFlow, we will design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. For this we need to think about how many inputs there are before determining the number of neurons and layers in the model. Later we will compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy. The steps involved in this process are shown below:
- Create a neural network model by assigning the number of input features and nodes for each layer using Tensorflow Keras.

![21](https://user-images.githubusercontent.com/23488019/158045613-390db02b-5960-4b11-a922-f74bb5e99b08.PNG)

- Create the first hidden layer and choose an appropriate activation function.
- If necessary, add a second hidden layer with an appropriate activation function.
- Create an output layer with an appropriate activation function.
- Check the structure of the model.
- Compile and train the model.

![22](https://user-images.githubusercontent.com/23488019/158045617-901a0d91-fcc9-422d-8dfb-587ba12d79a0.PNG)

- Create a callback that saves the model's weights every 5 epochs.
- Evaluate the model using the test data to determine the loss and accuracy.

![23](https://user-images.githubusercontent.com/23488019/158045619-890c5454-ffb6-4cf8-bb6c-4e0c6c233653.PNG)

- Save and export  results to an HDF5 file, and name it AlphabetSoupCharity.h5.
Hence, the neural network model using Tensorflow Keras performs the following steps:
- The number of layers, the number of neurons per layer, and activation function are defined 
- An output layer with an activation function is created 
- There is an output for the structure of the model 
- There is an output of the model’s loss and accuracy 
- The model's weights are saved every 5 epochs 
- The results are saved to an HDF5 file 


### 3. Deliverable 3: Optimize the Model
Using TensorFlow, we will optimize our model in order to achieve a target predictive accuracy higher than 75%. If we can't achieve an accuracy higher than 75%, then we will make at least three attempts to do so.

This deliverable can be achieved by doing any or all of the following steps:
1. Adjusting the input data to ensure that there are no variables or outliers that are causing confusion in the model, such as:
 - Dropping more or fewer columns.
 - Creating more bins for rare occurrences in columns.
 - Increasing or decreasing the number of values for each bin.
2. Adding more neurons to a hidden layer.
3. Adding more hidden layers.
4. Using different activation functions for the hidden layers.
5. Adding or reducing the number of epochs to the training regimen.


## Results :

#### A. Data Preprocessing - 
1. What variable(s) are considered the target(s) for your model?
   The variable that was considered as the target for our model was the IS_SUCCESSFUL column.

2. What variable(s) are considered to be the features for your model?
   The features of my model are:
   - AFFILIATION
   - APPLICATION_TYPE
   - ASK_AMT
   - CLASSIFICATION
   - INCOME_AMT
   - ORGANIZATION
   - SPECIAL_CONSIDERATIONS
   - STATUS
   - USE_CASE
   
3. What variable(s) are neither targets nor features, and should be removed from the input data?
   The "NAME" and "EIN" variables do not affect the success or the failure rate and hence are dropped out of the table.
   
   
#### B. Compiling, Training, and Evaluating the Model - 
1. How many neurons, layers, and activation functions did you select for your neural network model, and why?

2. Were you able to achieve the target model performance?


3. What steps did you take to try and increase model performance?


## Summary
Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.
