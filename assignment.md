This assignment can be completed in teams of three students. 
Exceptionally, teams of two will be accepted upon my approval.

To do this assignment, you have to use the python library Surprise, 
which provides implementations for the CF algorithms addressed in the course. 
Use ```random_state=22``` for splitting the model into the train and test sets. 
Use ```random_state=3``` for creating the SVD model.

You will have to submit a zip/rar file containing a document explaining 
the results obtained in the proposed tasks and the discussion of these results, 
as well as the source code. 
The usage of graphs to show the results obtained from the scripts will be 
positively assessed.

The assignment consists of the following tasks:

## 1. 
Given this data set (Movielens 100k) 
and the algorithm of K-NN explained in class for user-based CF:

### a)  (FAIT)
Find out the value for K<=100 that minimizes the MAE with 25% of missing ratings. 
Check all the K values with an increase of +1 up to 100. 
Make sure that you use the sufficient decimal precision to compare the values, 
because the differences between the MAEs will be very small.

### b) (FAIT)
Sparsity Problem: 
find out the value for K under the same conditions above 
that minimizes the MAE with 75% of missing ratings.

## 2. Mitigation of sparsity problem: (FAIT)
show how SVD (Funk variant) can provide a better MAE than user-based K-NN 
using the provided data set. 

 

## 3. Top-N recommendations: 
calculate the precision, recall, and F1 with different values for N (10..100) 
using user-based K-NN (with the best Ks)  and SVD. 
To do this, you must suppose that the relevant recommendations for a specific user 
are those rated with 4 or 5 stars in the data set. 
Perform the calculations for both 25% and 75% of missing ratings.

Explain why you think that the results reported in the three tasks make sense.

## 4. (Only for students of HCID master) 
Repeat the task 3 only for N=10 and 25% of missing ratings but 
using a different data set (Movielens 1M). 
Justify the differences between the values obtained with each data set 
for precision, recall and F1.



