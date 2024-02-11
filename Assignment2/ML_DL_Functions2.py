import numpy as np
# Dont add any more imports here!

# Make Sure you fill your ID here. Without this ID you will not get a grade!
def ID1():
    '''
        Personal ID of the first student.
    '''
    # Insert your ID here
    return 319149878
def ID2():
    '''
        Personal ID of the second student.
    '''
    # Insert your ID here
    return 000000000

def sigmoid(z):
  return 1 / (1 + np.exp(-z))
    
def cross_entropy(t, y):
  return -t * np.log(y) - (1 - t) * np.log(1 - y)


def get_accuracy(y, t):
  acc = 0
  N = 0
  for i in range(len(y)):
    N += 1
    if (y[i] >= 0.5 and t[i] == 1) or (y[i] < 0.5 and t[i] == 0):
      acc += 1
  return acc / N

def pred(w, b, X):
  """
  Returns the prediction `y` of the target based on the weights `w` and scalar bias `b`.

  Preconditions: np.shape(w) == (90,)
                 type(b) == float
                 np.shape(X) = (N, 90) for some N
  Postconditions: np.shape(y)==(N,)

  >>> pred(np.zeros(90), 1, np.ones([2, 90]))
  array([0.73105858, 0.73105858]) # It's okay if your output differs in the last decimals
  """
  z = np.dot(X, w) + b  # z=wX+b
  y = sigmoid(z)        # y=sigma(z)
  
  return y

def cost(y, t):
  """
  Returns the cost(risk function) `L` of the prediction 'y' and the ground truth 't'.

  - parameter y: prediction
  - parameter t: ground truth
  - return L: cost/risk
  Preconditions: np.shape(y) == (N,) for some N
                 np.shape(t) == (N,)
  
  Postconditions: type(L) == float
  >>> cost(0.5*np.ones(90), np.ones(90))
  0.69314718 # It's okay if your output differs in the last decimals
  """
  # Compute the cross-entropy loss
  L = -np.mean(t * np.log(y) + (1 - t) * np.log(1 - y))
  
  return L

def derivative_cost(X, y, t):
  """
  Returns a tuple containing the gradients dLdw and dLdb.

  Precondition: np.shape(X) == (N, 90) for some N
                np.shape(y) == (N,)
                np.shape(t) == (N,)

  Postcondition: np.shape(dLdw) = (90,)
           type(dLdb) = float
           return dLdw,dldb
  """
  # Your code goes here
  # we'll compute the derivative by the chain rule
  #epsilon = 1e-8  # Small value to avoid division by zero or close-to-zero values
  #y = np.clip(y, epsilon, 1 - epsilon)  # Clip y to avoid division by zero
    
  #dL_dy = -((t / y) - ((1 - t) / (1 - y))) 
  #dy_dz = y * (1 - y)
  #dz_dw = X
  #dz_db = 1
  
  #dLdw = np.dot(dL_dy * dy_dz, dz_dw)
  #dLdb = np.sum(dL_dy * dy_dz * dz_db)
  N = len(y)
  dLdw = np.dot(X.T, y-t)/N
  dLdb = np.mean(y-t)
  
  return (dLdw,dLdb)