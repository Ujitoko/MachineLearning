import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class LinearClassifier():
  N = 100 # number of points per class
  D = 2 # dimensionality
  K = 3 # number of classes
  X = np.zeros((N*K,D)) # data matrix (each row = single example)
  y = np.zeros(N*K, dtype='uint8') # class labels

  h = 100 # size of hidden layer
  W1 = 0.01 * np.random.randn(D,h)
  b1 = np.zeros((1,h))
  W2 = 0.01 * np.random.randn(h,K)
  b2 = np.zeros((1,K))

  step_size = 0
  reg = 1e-3

  losses = np.array([])
  accuracies = np.array([])
  
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
    plt.grid(True)
    plt.savefig("./"+"Data"+".png")

  def init_param(self):
                
    self.h = 100 # size of hidden layer
    self.W1 = 0.01 * np.random.randn(self.D,self.h)
    self.b1 = np.zeros((1,self.h))
    self.W2 = 0.01 * np.random.randn(self.h,self.K)
    self.b2 = np.zeros((1,self.K))

  def init_hyperparam(self):
    # some hyperparameters
    self.step_size = 1e-0
    self.reg = 1e-3 # regularization strength

  def grad_descent_loop(self):
    # gradient descent loop
    num_examples = self.X.shape[0]
    for i in range(550000):
      hidden_layer = np.maximum(0, np.dot(self.X, self.W1) + self.b1)
      scores = np.dot(hidden_layer, self.W2) + self.b2 

      # compute the class probabilities
      exp_scores = np.exp(scores)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True) # [N x K]
  
      # compute the loss: average cross-entropy loss and regularization
      correct_logprobs = -np.log(probs[range(num_examples),self.y])
#      print(correct_logprobs.shape)
      data_loss = np.sum(correct_logprobs)/num_examples
      reg_loss = 0.5*self.reg*np.sum(self.W1*self.W1) + 0.5*self.reg*np.sum(self.W2*self.W2)
      loss = data_loss + reg_loss
      if i % 1000 == 0:
        print("iteration %d: loss %f" % (i, loss))
  
        # compute the gradient on scores
        dscores = probs
        dscores[range(num_examples),self.y] -= 1
        dscores /= num_examples
  
        # backpropate the gradient to the parameters (W,b)
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)

        dhidden = np.dot(dscores, self.W2.T)
        dhidden[hidden_layer <= 0] = 0

        dW1 = np.dot(self.X.T, dhidden)
        db1 = np.sum(dhidden, axis=0, keepdims=True)

        dW2 += self.reg*self.W2
        dW1 += self.reg*self.W1
        # perform a parameter update
        
        self.W2 += -self.step_size * dW2
        self.b2 += -self.step_size * db2
        
        self.W1 += -self.step_size * dW1
        self.b1 += -self.step_size * db1

      if i % 10000 == 0:
        self.losses = np.append(self.losses, loss)
        self.accuracies = np.append(self.accuracies, self.evaluate())
        self.draw_classified_area(isDraw=False, isSave=True, idx=(i/10000))
    
  def evaluate(self):
    # evaluate training set accuracy
    hidden_layer = np.maximum(0, np.dot(self.X, self.W1) + self.b1)
    scores = np.dot(hidden_layer, self.W2) + self.b2
    predicted_class = np.argmax(scores, axis=1)
    print ('training accuracy: %.2f' % (np.mean(predicted_class == self.y)))

    return (np.mean(predicted_class == self.y))
    
  def draw_classified_area(self, isDraw=True, isSave=False, idx=None):
    # X
    n = 50
    N = n**2
    K = 3
    D = 2
    X_lattice = np.zeros((N*K,D))
    for i in range(n):
      for j in range(n):
        X_lattice[i*n+j] = np.c_[-1 + 2*i*(1/n), -1 + 2*j*(1/n)]

    hidden_layer = np.maximum(0, np.dot(X_lattice, self.W1) + self.b1)
    scores = np.dot(hidden_layer, self.W2) + self.b2
    predicted_class = np.argmax(scores, axis=1)
    plt.scatter(X_lattice[:, 0], X_lattice[:, 1], c=predicted_class, s=100, cmap=plt.cm.Spectral, linewidth="0")
    plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, s=40, cmap=plt.cm.Spectral)
    plt.xlim(-1.0,1.0)
    plt.ylim(-1.0,1.0)
    plt.grid(True)
    
    if isDraw==True:
      plt.show()
    if isSave==True:
#      plt.savefig("./"+str(idx)+".gif")
      plt.savefig("./" + str(idx) + ".png")
#      self.ims.append(im)

    
### main
if __name__ == "__main__":

  LC = LinearClassifier()
  LC.init_param()
  LC.init_hyperparam()
  LC.grad_descent_loop()
#  LC.evaluate()
#  LC.draw_classified_area()

  plt.figure()
  plt.title('Accuracy')
  plt.ylim(0.0, 1.0)
  plt.plot(LC.accuracies)
  plt.savefig("./"+"Accuracy"+".png")
#  plt.show()

  plt.figure()
  plt.title('Loss')
  plt.ylim(0.0, 1.2)
  plt.plot(LC.losses)
  plt.savefig("./"+"Loss"+".png")
#  plt.show()
  
#  ani = animation.ArtistAnimation(LC.fig, LC.ims, interval=100)
 # ani.save("hoge.gif")
