# CNN - Walk through of PyTorch internals 
> A quick run down on the internals of the a simple CNN right from the fetching of the data to the training loop. This is a quick reference on ensuring every component is tied correctly and how the abstractions work.
Knowing the sequence correctly can be incredibly useful in knowing what each component contributes to the grand scheme of things.



```python
%load_ext autoreload
%autoreload 2

%matplotlib inline
```

```python
# export
%reload_ext autoreload
from pathlib import Path
from IPython.core.debugger import set_trace
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.functional import F
from torch import nn
from torch import optim

```

```python
MNIST_URL = 'http://deeplearning.net/data/mnist/mnist.pkl'
```

### Steps

1. Get Data
2. Create Datasets
3. Specify batch size `bs`, input shape `m` and output_shape `c` (number of output categories)
5. Define Loss function
4. Create a DataLoaders
5. Create a Databunch
6. Create model 
7. Specifiy optimizer
8. Create fit function 
9. Create learner
10. Create callbacks

### 1. Get the data and convert it to tensors

```python
def get_data():
    path = datasets.download_data(MNIST_URL, ext='.gz')
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    return map(tensor, (x_train, y_train, x_valid, y_valid))
```

```python
x_train, y_train, x_valid, y_valid = get_data()
```

### 2. Create Datasets or using the default PyTorch's inherited Datasets

```python
class DataSet():
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]
    def __repr__(self): return f"{self.x.shape}, {self.y.shape}"
```

```python
train_ds = DataSet(x_train, y_train)
valid_ds = DataSet(x_valid, y_valid)
```

### 3. Specify batchsize

```python
bs=16
```

### Loss Function

```python
loss_func = F.cross_entropy
```

### 4. DataLoader

```python

def collate(b):
    xs,ys = zip(*b)
    return torch.stack(xs),torch.stack(ys)

def get_dls(train_ds, valid_ds, bs):
    return (DataLoader(train_ds, bs, sampler=RandomSampler(train_ds), collate_fn=collate), 
            DataLoader(valid_ds, bs, sampler=SequentialSampler(valid_ds), collate_fn=collate))

train_dl, valid_dl = get_dls(train_ds, valid_ds, bs); 
```

### 5. DataBunch

```python
class DataBunch():
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c
    
    @property
    def train_ds(self):
        return self.train_dl.dataset
    
    @property
    def valid_ds(self):
        return self.valid_dl.dataset
```

```python
db = DataBunch(train_dl, valid_dl, c=10); db.train_ds
```




    torch.Size([50000, 784]), torch.Size([50000])



### 6. Create model

```python
def get_model(db:DataBunch, lr:float=0.5, num_hidden_layers:int=50) -> (nn.Sequential, optim.SGD):
    nh = num_hidden_layers
    m = db.train_ds.x.shape[1]
    c = db.c
    model = nn.Sequential(
        nn.Linear(m, nh),
        nn.ReLU(),
        nn.Linear(nh, c)
    )
    
    return model, optim.SGD(model.parameters(), lr=lr)
```

```python
model, opt = get_model(db)
model
```




    Sequential(
      (0): Linear(in_features=784, out_features=50, bias=True)
      (1): ReLU()
      (2): Linear(in_features=50, out_features=10, bias=True)
    )



### 7. Create Learner

```python
class Learner():
    def __init__(self, model, opt, loss_func, db):
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.data = db
```

```python
learner = Learner(model, loss_func=loss_func, opt=opt,db=db); learner
```




    <__main__.Learner at 0x7fa1e9a6fd50>



### 8. Create training loop and metric (accuracy in this case)

```python
def accuracy(out, yb, debug=False):
    if debug: print(f'output: {out}, yb: {yb}')
    return (torch.argmax(out, dim=1)==yb).float().mean()

def fit(epochs, learn):
    for epoch in range(epochs):
        print(f'Epoch #{epoch}')
        learn.model.train()
        for xb, yb in learn.data.train_dl:
            loss = learn.loss_func(learn.model(xb), yb)
            loss.backward()
            learn.opt.step()
            learn.opt.zero_grad()
            
        learn.model.eval()
        with torch.no_grad():
            total_loss, total_accuracy = 0.,0.
            for xb, yb in learn.data.valid_dl:
                total_loss += learn.loss_func(learn.model(xb), yb)
                total_accuracy += accuracy(learn.model(xb), yb)
            nv = len(learn.data.valid_dl)
            print(epoch, total_loss/nv, total_accuracy/nv)
    return (total_loss/nv, total_accuracy/nv)
```

```python
loss, acc = fit(3, learner)
```

    Epoch #0
    0 tensor(0.1875) tensor(0.9610)
    Epoch #1
    1 tensor(0.2076) tensor(0.9576)
    Epoch #2
    2 tensor(0.1788) tensor(0.9616)

