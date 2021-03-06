import numpy as np
import matplotlib.pyplot as plt

class LinearClassifier():
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels

  W = 0.01 * np.random.randn(D,K)
  b = np.zeros((1,K))

  step_size = 0
  reg = 1e-3

  def __init__(self):
    for j in range(self.K):
      ix = range(self.N*j,self.N*(j+1))
      r = (1 + 0.05*np.random.randn(self.N)) * ((j+1)/(self.K + 1)) 
      t = np.linspace(j*4,(j+1)*4,self.N) + np.random.randn(self.N)*0.2 # theta
      self.X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
      self.y[ix] = j
      print(j)

    # lets visualize the data:
    plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40, cmap=plt.cm.Spectral)
#    plt.title("input data")
 #   plt.xlabel("x axis")
  #  plt.ylabel("y axis")
    plt.grid(True)
    plt.show()

  def init_param(self):
    self.W = 0.01 * np.random.randn(self.D,self.K)
#    print(self.W)
    self.b = np.zeros((1,self.K))

  def init_hyperparam(self):
    # some hyperparameters
    self.step_size = 1e-0
    self.reg = 1e-3 # regularization strength

  def grad_descent_loop(self):
    # gradient descent loop
    num_examples = self.X.shape[0]
    for i in range(300):
      # evaluate class scores, [N x K]
      scores = np.dot(self.X, self.W) + self.b 
  
      # compute the class probabilities
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
      # compute the loss: average cross-entropy loss and regularization
      correct_logprobs = -np.log(probs[range(num_examples),self.y])
#      print(correct_logprobs.shape)
      data_loss = np.sum(correct_logprobs)/num_examples
      reg_loss = 0.5*self.reg*np.sum(self.W*self.W)
      loss = data_loss + reg_loss
      if i % 10 == 0:
        print("iteration %d: loss %f" % (i, loss))
  
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),self.y] -= 1
        dscores /= num_examples
  
        # backpropate the gradient to the parameters (W,b)
        dW = np.dot(self.X.T, dscores)
        db = np.sum(dscores, axis=0, keepdims=True)
      
        dW += self.reg*self.W # regularization gradient
  
        # perform a parameter update
        self.W += -self.step_size * dW
        self.b += -self.step_size * db

  def evaluate(self):
    # evaluate training set accuracy
    scores = np.dot(self.X, self.W) + self.b
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == self.y)))

  def draw_classified_area(self):
    # X
    n = 50
    N = n**2
    K = 3
    D = 2
    X_lattice = np.zeros((N*K,D))
    for i in range(n):
      for j in range(n):
        X_lattice[i*n+j] = np.c_[-1 + 2*i*(1/n), -1 + 2*j*(1/n)]

    scores = np.dot(X_lattice, self.W) + self.b
    predicted_class = np.argmax(scores, axis=1)
    plt.scatter(X_lattice[:, 0], X_lattice[:, 1], c=predicted_class, s=100, cmap=plt.cm.Spectral, linewidth="0")
    plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(-1.0,1.0)
    plt.ylim(-1.0,1.0)
    plt.grid(True)
    plt.show()
### main
if __name__ == "__main__":

  LC = LinearClassifier()
  LC.init_param()
  LC.init_hyperparam()
  LC.grad_descent_loop()
  LC.evaluate()
  LC.draw_classified_area()
  
