import numpy as np
import matplotlib.pyplot as plt



def init(N, D, K, X, y):

  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels
  for j in range(K):
    ix = range(N*j,N*(j+1))
    r = np.linspace(0.0,1,N) # radius
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
    X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j
    print(j)
    # lets visualize the data:
    
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
  plt.title("input data")
  plt.xlabel("x axis")
  plt.ylabel("y axis")
  plt.grid(True)
  plt.show()

#Train a Linear Classifier

def init_param(D, K, W, b):
  W = 0.01 * np.random.randn(D,K)
  b = np.zeros((1,K))

def init_hyperparam(step_size, reg):
  # some hyperparameters
  step_size = 1e-0
  reg = 1e-3 # regularization strength

def grad_descent_loop(N, D, K, X, y, W, b, step_size, reg):
  # gradient descent loop
  num_examples = X.shape[0]
  for i in range(300):
    
    # evaluate class scores, [N x K]
    scores = np.dot(X, W) + b 
  
    # compute the class probabilities
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
    # compute the loss: average cross-entropy loss and regularization
    corect_logprobs = -np.log(probs[range(num_examples),y])
    data_loss = np.sum(corect_logprobs)/num_examples
    reg_loss = 0.5*reg*np.sum(W*W)
    loss = data_loss + reg_loss
    if i % 10 == 0:
      print("iteration %d: loss %f" % (i, loss))
  
      # compute the gradient on scores
      dscores = probs
      dscores[range(num_examples),y] -= 1
      dscores /= num_examples
  
      # backpropate the gradient to the parameters (W,b)
      dW = np.dot(X.T, dscores)
      db = np.sum(dscores, axis=0, keepdims=True)
      
      dW += reg*W # regularization gradient
  
      # perform a parameter update
      W += -step_size * dW
      b += -step_size * db

def evaluate(X, W, b, y):     
  # evaluate training set accuracy
  scores = np.dot(X, W) + b
  predicted_class = np.argmax(scores, axis=1)
  print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))



### main
if __name__ == "__main__":
  N = 0
  D = 0
  K = 0
  X = np.array(1)
  y = 0
  init(N, D, K, X, y)
  print(X)
  W = 0
  b = 0
#  init_param(D, K, W, b)

  step_size = 0
  reg = 0
#  init_hyperparam(step_size, reg)

#  grad_descent_loop(N, D, K, X, y, W, b, step_size, reg)

#  evaluate(X, W, b, y)

  
