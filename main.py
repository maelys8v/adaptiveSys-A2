#from collections import defaultdict
#from pprint import pprint

import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset, SVD
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

todo_1a = False
todo_1b = False
todo_2 = True
# ---------------------------------------------------------------
def compute_MAE_for_100k_knn(trainset, testset):
    MAE = []
    bestMAE= float("inf")
    k_bestMAE = -1
    for k in range (1,101):
        print(k)
        algo = KNNWithMeans(k=k, sim_options={"name": "pearson","user_based":True}) # neighbours are users
        algo.fit(trainset)
        MAE_k = mae(algo.test(testset), verbose=False)
        MAE.append(MAE_k)
        if MAE_k < bestMAE:
            bestMAE, k_bestMAE = MAE_k, k

    print("---------------------------------------------------------------")
    print("Best K:", k_bestMAE)
    print("Best MAE:", f"{bestMAE:.10f}")
    return bestMAE, k_bestMAE

def compute_MAE_svd(trainset, testset):
    algo = SVD(random_state=3)
    algo.fit(trainset)
    MAE_svd = mae(algo.test(testset), verbose=False)
    print(f"SVD MAE : {MAE_svd:.10f}")
    return MAE_svd
# ---------------------------------------------------------------
df = pd.read_csv(
    "u_data.csv",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"]
)

reader = Reader(rating_scale=(1,5)) # because the ratings are between 1 and 5
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]],reader)

# PART 1a ------------------------------------------------------
if(todo_1a):

    trainset1a, testset1a = train_test_split(
        data,
        test_size=0.25,
        random_state=22
    )

    bestMAE1a, k_bestMAE1a = compute_MAE_for_100k_knn(
        trainset1a,
        testset1a
    )
    # Results 1a :
    # k_bestMAE1a, bestMAE1a = 75, 0.7446623579

# PART 1b ------------------------------------------------------
if todo_1b:
    trainset1b, testset1b = train_test_split(
        data,
        test_size=0.75,
        random_state=22
    )

    bestMAE1b, k_bestMAE1b = compute_MAE_for_100k_knn(
        trainset1b,
        testset1b
    )
    # Results 1b :
    # k_bestMAE1b, bestMAE1b = 61, 0.8125340341

# PART 2 --------------------------------------------------------------------------------------------------------
if todo_2:
    trainset2_25, testset2_25 = train_test_split(
        data,
        test_size=0.25,
        random_state=22
    )

    trainset2_75, testset2_75 = train_test_split(
        data,
        test_size=0.75,
        random_state=22
    )

    MAE_25 = compute_MAE_svd(
        trainset2_25,
        testset2_25
    )
    # MAE25 = 0.7387569723

    MAE_75 = compute_MAE_svd(
        trainset2_75,
        testset2_75
    )
    # MAE75 = 0.7735654307


