<p align= "center">
<a href="https://colab.research.google.com/github/martian1231/KNNFromScratch/blob/main/K_nearest_neighbour_from_scratch(classification%20and%20regression).ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab"></a>
</p>

# K Nearest Neighbour (KNN, neighbourhood based machine learning algorithm)
**KNN is a non-parametric statistical model (algorithm) used to solve variety of types of machine learning problems which includes supervised learning (classification, regression) and unsupervised learning also called K means clustering.**

### KNN in brief

* It is simple to understand, Intuition behind KNN is really simple.
* It uses basic distance metric to compute distance between two point. ie  Euclidean or Manhattan.
*  K in KNN is "Number of nearest neighbours" you want to consider to make a decision (ie, computing 3 nearest neighbours, taking majority vote of there NN classes and assigning majority class to the query point, in the case of classification.
*  K should be odd ie (3, 5, etc) in order to avoid conflict between NN classes. ie If K = 4 and NN classes happen to be 0, 0, 1 ,1, hence majority voting won't work (unless we assign either of the class randomly).
*  Use cross validation to choose best value of K (using Grid search or random search).
*  It is non-parametric model, It does not assume any distribution about the data unlike linear regression.
*  No training required: KNN is called **Lazy Learner** (Instance based learning). It does not learn anything in the training period. This makes the KNN algorithm much faster than other algorithms(train time) that require training e.g. SVM, Linear Regression etc (this is the reason why it as O(1) train time).
* Since the KNN algorithm requires no training before making predictions, new data can be added seamlessly which will not impact in the accuracy of the algorithm.
* KNN is very easy to implement. There are only two parameters required to implement KNN i.e. the value of `K` and the distance function (e.g. Euclidean or Manhattan etc.)
* Does not work well with large dataset: In large datasets, the cost of calculating the distance between the new point and each existing points is huge which degrades the performance of the algorithm.
* Does not work well with high dimensions: The KNN algorithm doesn't work well with high dimensional data because with large number of dimensions, it becomes difficult for the algorithm to calculate the distance in each dimension. (This issue could be solved using space partitioning algorithms like KD tree, ball tree, LSH, etc that reduces the search spaces).
* Need feature scaling: We need to do feature scaling (standardization and normalization) before applying KNN algorithm to any dataset as it is distance based algorithm (like linear regression and logistic regression). If we don't do so, KNN may generate wrong predictions.
* Sensitive to noisy data, missing values and outliers: KNN is sensitive to noise in the dataset. We need to manually impute missing values and remove outliers before running KNN.

### Assumptions of KNN

* KNN assumes, data that are similar should be close to each other in the feature space, As it makes prediction based on the neighbour of a query point.

* KNN assumes data is normalized. As KNN compares the data points based on some distance metric, we should make sure comparisons are fair and all the features are in same unit. Height in cms and weight in KGs would be different scale comparison as height will dominate the age.

* Low k-value is sensitive to outliers and a higher K-value is more resilient to outliers as it considers more voters for final prediction.

### KNN for classification

1. Choose value of K, How many nearest neighbour you need to take into account before making a decision (before assigning class to a query point). **K should be odd as to keep from conflicting.**
2. Compute distance of query point from all other points in the dataset (could use Euclidean or Manhattan distance).

3. Select K point which is minimum distance from query point.

4. Take majority vote across K nearest neighbour classes from a query point.

5. Assign majority class as class label to a query point.

**To find best value of K we perform K fold cross validation using different-different value of K then select K which yield maximum accuracy score. Performance Metric of KNN classifier is usually ACCURACY SCORE**.

### Objective function of KNN classifier

<img src="https://i.imgur.com/H9y35eF.png" alt="img" style="zoom:50%;" />

What this says it that the predicted class t^ for a point x^ is equal to the class C which maximizes the number of other points xi that are in the set of k nearby points Nk({x},x^) that also have the same class, measured by δ(ti,C) which is 1 when xi is in class C, 0 otherwise.

The advantage of writing it this way is that one can see how to make the objective "softer" by weighting points by proximity. Regarding "training," there are no parameters here to fit. But one could tune the distance metric (which is used to define Nk) or the weighting of points in this sum to optimize some additional classification objective.

[Does KNN have a loss function?](https://stats.stackexchange.com/questions/420416/does-knn-have-a-loss-function)

![KNN_image](https://cambridgecoding.files.wordpress.com/2016/01/knn2.jpg)

### Weighted KNN
  * Weighted KNN classifier assigns more weight (give more priority) to point which are much close to query point than points which are further away.

  * After getting K nearest neighbour point from the query point we apply inverse function. Inverse function reverse/ inverse/ undo the operation of applied function. In KNN context when distance is low inverse function will return high value and vice-versa.

  * Simplest inverse function is `1/x`. Where `x` is the distance from the query point to training point P. As distance increases `x` value of inverse function will decrease and vice-versa. We will use this inverse function as a weight function that assigns more weight to distance which are closer to the query point.

  * After computing weights of all K nearest points, we sum up their weights based on their corresponding classes. eg
    _K nearest classes = [1, 0, 0], dist = [0.1, 0.5, 0.55], weights = [10, 2, 1.81], summing weights among each class = {1: 10, 0: 3.81}. And hence we declare class label of the query point as 1_.

    <img src="https://i.stack.imgur.com/N3I5R.png" alt="Weighted KNN classifier" style="zoom:50%;" />

       **Problem with vanilla KNN classifier**
      _In above image, query point `X` belongs to green class, but majority vote will assign it as blue_. Weighted KNN will assign more weight based on distance,more weight to green point which is close to query point `X` than to both the blue point which is further away from the query point `X` and hence weighted KNN will declare the query point `X` as green.

### Failure case of KNN classifier

* KNN doesn't work much good when dimensionality of data is high due to curse of dimensionality and because of huge computation cost.
* If query point is far away from the training point or data is random, conclusions by KNN classifier might be wrong.

#### Bias-Variance trade-off in KNN classifier

* Smaller value of `K` leads to overfitting because we are not using surrounding information but just one closest neighbour to make decision (no ensembling).
* Huge value of `K` leads to underfitting and favour majority class if classes are imbalanced.
* Robust to outlier if value of `K` is large enough.

![k-nearest neighbor classification](https://cse630.artifice.cc/images/effect-of-k.png)

#### KNN for regression

KNN for regression works same as classification, But rather than taking majority vote, we take mean or median values of `K` nearest neighbours.

<img src="https://image.slidesharecdn.com/knn-170219223104/95/k-nearest-neighbor-presentation-10-638.jpg?cb=1487543544" alt="K Nearest Neighbor Presentation" style="zoom:50%;" />

#### Making KNN efficient (reducing the search for K neighbours)

In typical KNN, we computer distance of all points in training from the query point and select `K` nearest points and take the majority vote ( in the case of classification) and mean or median (in the cause of regression).

There is a way to reduce the search space using "**spatial index**" data structure and **space partitioning** algorithms.

A **spatial index** is a data structure that allows for accessing a **spatial** object efficiently. It is a common technique used by **spatial** databases. Without **indexing**, any search for a feature would require a "sequential scan" of every record in the database, resulting in much longer processing time.

Here is list of "spatial index" data structure:

- [Geohash](https://en.wikipedia.org/wiki/Geohash)
- [HHCode](https://en.wikipedia.org/wiki/HHCode)
- [Grid (spatial index)](https://en.wikipedia.org/wiki/Grid_(spatial_index))
- [Z-order (curve)](https://en.wikipedia.org/wiki/Z-order_(curve))
- [Quadtree](https://en.wikipedia.org/wiki/Quadtree)
- [Octree](https://en.wikipedia.org/wiki/Octree)
- [UB-tree](https://en.wikipedia.org/wiki/UB-tree)
- [R-tree](https://en.wikipedia.org/wiki/R-tree): Typically the preferred method for indexing spatial data.[*[citation needed](https://en.wikipedia.org/wiki/Wikipedia:Citation_needed)*] Objects (shapes, lines and points) are grouped using the [minimum bounding rectangle](https://en.wikipedia.org/wiki/Minimum_bounding_rectangle) (MBR). Objects are added to an MBR within the index that will lead to the smallest increase in its size.
- [R+ tree](https://en.wikipedia.org/wiki/R%2B_tree)
- [R* tree](https://en.wikipedia.org/wiki/R*_tree)
- [Hilbert R-tree](https://en.wikipedia.org/wiki/Hilbert_R-tree)
- [X-tree](https://en.wikipedia.org/wiki/X-tree)
- [kd-tree](https://en.wikipedia.org/wiki/Kd-tree)
- [m-tree](https://en.wikipedia.org/wiki/M-tree) – an m-tree index can be used for the efficient resolution of similarity queries on complex objects as compared using an arbitrary metric.
- [Binary space partitioning](https://en.wikipedia.org/wiki/Binary_space_partitioning) (BSP-Tree): Subdividing space by hyperplanes.

### Train Time, Test Time And Runtime Complexity

* There is no training in KNN and hence training time is constant `O(1)`,
* If we are using space partitioning algorithm, creating the data structure could take `O(Nlog(N))` time complexity (or `N*d` if number of features is huge), which could be consider as train time.
*  Test time is `O(N)` for each query point, because we need to compute distance to each and every point in the dataset.
* Test time cold be reduced to `O(log(N))` per query point if we are using space partitioning algorithm.
* Space complexity is `O(N*d)` because we need to store all the data points to compute distance (with or without partitioning algorithm).

### KEY TAKEAWAYS

* KNN is a neighbourhood based algorithm.
* As it is a distance based algorithm, feature normalization should be done.
* Sensitive to feature scale.
* It is non parametric, assumes nothing about distribution of underlying data unlike other distance based algorithm like linear and logistic regression.
* Bias-Variance is controlled by parameter `K` which should be chosen after cross validation.
* Higher value of `K` results in underfitting and lower value of `K` results in overfitting.
* As `K` approaches `N` decision gets biased towards majority class (if dataset is imbalanced) or each class is equally likely (if dataset is balanced).
* Takes majority voting of `K` nearest neighbours in the case of classification and mean/ median in the case of regression.
* Use weighted KNN to weight point opposite of distance to the query point.
* Use space partitioning algorithms to fast search for nearest neighbours (must use if there are lots of query points).
* Train time is `O(1)` ( `O(Nlog(N))`if creating space partitioning algorithm to fasten neighbour search).
* Test time is `O(N*d)` (`O(log(N))`if using space partitioning algorithm).
* Space complexity is `O(N*d)`.
* Dimensionality reduction technique could be used to fasten KNN at test time.

### References, Citations And Sources

[AppliedAICourse](https://www.appliedaicourse.com/lecture/11/applied-machine-learning-online-course/2927/k-nearest-neighbours-geometric-intuition-with-a-toy-example/3/module-3-foundations-of-natural-language-processing-and-machine-learning)

[Space partitioning](https://en.wikipedia.org/wiki/Space_partitioning)

[Spatial Database](https://en.wikipedia.org/wiki/Spatial_database)

[Google Image Search](images.google.com)

[Does KNN have a loss function?](https://stats.stackexchange.com/questions/420416/does-knn-have-a-loss-function)
