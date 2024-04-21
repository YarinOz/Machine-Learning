import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 000000000
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  theta_star = np.linalg.inv(X.T @ X) @ X.T @ y
  return theta_star

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
    # Make predictions on the test set
  predictions = model.predict(X)

    # Calculate the number of correct predictions
  correct_predictions = sum(1 for true, pred in zip(s, predictions) if true == pred)

    # Calculate the total number of predictions
  total_predictions = len(s)

    # Calculate accuracy as a percentage
  accuracy = (correct_predictions / total_predictions) * 100

  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-0.06621319807650804, 0.025835996988919085, -0.042804624665499114, 0.009218260687539084, -0.03176535981927929,
  -0.0021894613669925043, 0.08063438756675513, -0.020886033296341658, 0.03713216385446223, -0.011286082167903872, 0.025911144635253652,
  0.023877945266988478, 0.08840549246120818, 0.13852364432102166, 0.789411497365006, 0.040593816175079636, 0.01642821941853255,
  0.01696920527371287, 0.006018778719651471, 0.0010774454177221143, 0.03946285086274321, 0.03646167126059875, 0.01069012653322952,
  -0.018237704979649383, -0.027254867443904117, 0.02450038867935066, -0.03841682640435068, -0.0314986833014394]


def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return 1.160753384931821e-17

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.09297135010686193, -0.2398978956726984, 0.18322805661793923, -0.024588029402329663, -0.16910531697091466,
  -0.9308424791743946, -0.08287336935049912, 0.1388666203150479, 0.09052822432077194, 0.2211329304394299, -0.5061792794569571,
  0.061164727761461396, -0.13350567116325454, 1.2212778648076092, 2.766605385219423, -0.48063912280938836, -0.09785664762220736,
  0.09164735446242597, -0.20912513019605414, -0.12239697175607853, 0.18926779076016648, -0.1649639228353053, 0.21429410381825892,
  -0.21712829147691137, -0.43972981453271676, -0.005491205107636177, -0.01630032008191021, 0.2580040836789256]]


def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.01224814]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [-1 , 0]