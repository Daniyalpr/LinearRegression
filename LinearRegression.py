import numpy as np

class Model():
  """This model assumes that the input is like this:
  The label is 1D matrix(a vector) for example y = [1, 2, 3 , 0, 0.1].
  The input data is 2D matrix which every column corresponds to a feature and every row is a
  diffirent data; for example data[1, 0] represents the first feature of second
  data
  """
  def __init__(self, n_features):
    self.w = np.random.rand(n_features)
    self.b = np.random.rand()
    self.n_features = n_features

  def pred(self, data):
    z = np.dot(data, self.w.reshape(self.n_features, 1)) + self.b
    return z.reshape(data.shape[0])

  def fit(self, data, label, epochs, lr=0.01):
    n = data.shape[0]
    for epoch in range(1, epochs+1):
      error = self.pred(data) - label

      #dw = np.dot(error, data)/n
      dw = (data.T @ error) / n
      self.w -= lr * dw

      db = np.mean(error)
      self.b -= lr * db

      MSE = np.mean(error**2)/2
      percent = sum(error)/sum(label) * 100
      percent = abs(percent)
      if epoch % 10 == 0:
        print(f"epoch {epoch} | error --> {percent:.2f}% | MSE ---> {int(MSE)} | RMSE ---> {int(np.sqrt(MSE))}" )

  def __call__(self, features):
    return self.pred(features)
    
