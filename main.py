#from collections import defaultdict
#from pprint import pprint

import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

df = pd.read_csv(
    "u_data.csv",
    sep="\t",
    header=None,
    names=["user_id", "item_id", "rating", "timestamp"]
)

# PART 1a
# do I have sufficient decimal precision to compare the values, because the differences between the MAEs will be very small ?

reader = Reader(rating_scale=(1,5)) # because the ratings are between 1 and 5
data = Dataset.load_from_df(df[["user_id", "item_id", "rating"]],reader)
trainset, testset = train_test_split(data, test_size=0.25, random_state=22)

def compute_MAE_for_100k():
    MAE = []
    bestMAE= float("inf")
    k_bestMAE = -1
    for k in range (1,101): # it does not use 101 ??? I need the values from 1 to 100
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

bestMAE, k_bestMAE = compute_MAE_for_100k()
# bestMAE, k_bestMAE = 75, 0.7446623579








