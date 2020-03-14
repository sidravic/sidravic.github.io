# Mean Absolute Deviation
> Minor tibdit on what mean absolute deviation is. 


<br>
This is a trivial tidbit that is useful to have as a quick glance.

The mean absolute deviation which is essentially just 

```python
t = torch.tensor([1, 2,3, 5, 68 ]).float(); 
m = t.mean(); 
(t-m).abs().mean()
```

is less susceptible to outliers than standard deviation.  

```python
%load_ext autoreload
%autoreload 2
%matplotlib inline
%reload_ext autoreload
from torch import tensor
```

```python
t = torch.tensor([1, 2,3, 5, 68 ]).float(); t
```
    tensor([ 1.,  2.,  3.,  5., 68.])

```python
m = t.mean(); m
```

    tensor(15.8000)


### Variance and Standard deviation
<br>
Variance

$$
  \frac{\sum(x − x_{mean})^2}{(n)}
$$

<br>
Standard deviation

$$
σ =\sqrt\frac{\sum(x −x_{mean})^2}{n}
$$

```python
variance = (t-m).pow(2).mean()
mean_absolute_deviation = (t-m).abs().mean()
standard_deviation = (t-m).pow(2).mean().sqrt()
variance, mean_absolute_deviation, standard_deviation
```




    (tensor(682.9600), tensor(20.8800), tensor(26.1335))



### Variance in a different form

```
(t-m).pow(2).mean() == (t*t).mean() - (m*m)
```

```python
variance2 = (t*t).mean() - (m*m)
torch.allclose(variance, variance2)
```




    True


