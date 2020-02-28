# Basic Stats Cheatsheet

## Variance 

1. [Youtube Video](https://www.youtube.com/watch?v=JIIXQaMXBVM)
2. Determines how far is each data point from the mean. 
3. If most points were closer to the mean it means the variances is lower  

$$
  \frac{\sum(x − x_{mean})^2}{(n − 1)}
$$

 
 
<br>
<br>
 
## Standard Deviation 

 

1. Is the positive square root of the variance 
2. $σ =\sqrt\frac{\sum(x −x_{mean})^2}{n}$
 
<br>
<br>


## Covariance 

 

1. [Youtube video](https://www.youtube.com/watch?v=xGbpuFNR1ME)
2. Used to analyze linear relationships between 2 variables. How do these behave as pairs?  
3. A positive value indicates a direct increasing linear relationship. If one goes up the other goes up 
4. A negative value indicates an inverse relationship. If one goes up the other goes down. 
5. Difference between Covariance and Correlation 
   1. Covariance determines the type of association not the strength. It only speaks the direction of the direction. The correlation talks about the strength of the relationship 

 
<br>
<br>

## Sample Covariance 
 

$$ \frac{\sum(x  −x`)(y −y`)}{(n-1)} $$
 


Slope of a line m is essentially covariance because b in y = mx + b 

$$
    m = \frac{\sum(x −x_{mean})(y −y_{mean})}{n −1}
$$

 
<br>
<br>

## Correlation 

 

1. Correlation is always between -1 and 1 
2. Correlation is standardized thus comparable. 
3. Covariance is not standardized just direction 
4. Suprious correlation: Two completely unrelated factors that seem to have mathematical correlation but have no sensible correlation in real life. Dog bars vs moon's phase 

<br>
<br>

## Pearson correlation coefficient = r 

And 

$$
r = \frac{Covariance(x, y)}{standard deviation(x) ∗ standard deviation(y)}
$$ 

 
$$
r = Pearson coefficient =\frac{\sum(x −x_{mean}) (y −y_{mean})}{(n−1)∗Sx ∗Sy }
$$ 

 

Where $Sx$ and $Sy$ are 

$$ 
    Sx=\frac{\sum(x −xmean)}{2√n −1}
$$
 
$$
    Sy=\frac{\sum(y −ymean)}{2√(n −1)}
$$ 
 

$r$ is the coefficient of correlation 
When $n$=number of elements in consideration 
 

 

 

 
 