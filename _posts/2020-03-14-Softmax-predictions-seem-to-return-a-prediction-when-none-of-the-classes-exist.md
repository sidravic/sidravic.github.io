# Softmax - Why does the classifier say an image contains a category when it actually doesn't


This is something I struggled to understand when i started building my first classifiers and it deals with understanding what softmax actually does. 

One of the things I ended up doing was introducing noise in the form of categories I wanted the classifier to ignore which in turn affected the accuracy.

<br>

## Softmax

The job of is to predict if one of the categories is present in the input image. However, softmax will return a likelihood for each category's presence in the image. So it's easy to assume the category with the highest value is present in the image. 
However, we know that's not true. 

Since we know softmax is 

$$
\hbox{softmax(x)}_{i} = \frac{e^{x_{i}}}{e^{x_{0}} + e^{x_{1}} + \cdots + e^{x_{n-1}}} 
$$

It's always going to add up to 1. 

The presence of a category can be determined by using a binomial approach instead.

$$ b=\frac{e^x}{1+e^x}$$

to determine if the category is actually present and is more representative of the ground truth.

![system schema](/images/softmax_images/softmax_binomial.png)

Understanding the image:

1. Image1, indicates via softmax that the category is most likely a fish, with a possibility of a building in the image.
2. For image 2, while the activations are significantly different but the softmax computation seems to be identical to image1

The binomial approach provides more clarity into what could actually be present.

1. Image1 says that there could be a fish along with a building and possibly a cat.
2. Image2 could possibly have a a fish in it but it but we can't be certain about anything.