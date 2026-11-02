from collections import defaultdict
#from pprint import pprint

import matplotlib
matplotlib.use("Agg")   # headless backend: we save graphs to files, no window needed
import matplotlib.pyplot as plt

import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset, SVD
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

todo_1a = False
todo_1b = False
todo_2 = False
todo_3 = False
todo_4 = False
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

# ===============================================================
# PART 3 HELPERS: Top-N evaluation (Precision@N, Recall@N, F1@N)
# ===============================================================

def precision_recall_at_n(predictions, n=10, threshold=4):
    """
    Compute Precision@N and Recall@N for each user.

    Definitions used here (standard information-retrieval ones):
      - An item is "relevant" for a user if its TRUE rating is >= threshold.
        Here threshold=4, so only 4- and 5-star ratings count as relevant
        (this is exactly what the assignment asks for).
      - For each user we sort the PREDICTED ratings (estimated) from highest
        to lowest and keep the top N items. Those are the "recommendations".
      - Hits         = relevant items that appear in that top N.
      - Precision@N  = hits / min(N, number of test items for that user)
                       (we divide by how many items we could actually rank,
                        which can be < N for users with few test items)
      - Recall@N     = hits / total number of relevant items for that user

    NOTE: we only rank items that are in the test set. That is the usual way
    to evaluate on a rating dataset, because we have no ground truth for
    items the user never rated.
    """
    # Group, for every user, all (estimated_rating, true_rating) pairs.
    user_est_true = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating, highest first -> this is the ranking.
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items for this user (true rating 4 or 5).
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)

        # Number of relevant items among the top N recommendations.
        n_rel_and_rec = sum(
            true_r >= threshold
            for _, true_r in user_ratings[:n]
        )

        # Precision: relevant in top N / how many we recommended.
        precisions[uid] = n_rel_and_rec / min(n, len(user_ratings))

        # Recall: relevant in top N / all relevant items for this user.
        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0.0

    return precisions, recalls


def top_n_metrics(predictions, n_values, threshold=4):
    """
    For every N in n_values, average Precision@N / Recall@N over users and
    derive F1@N. Returns a dict: N -> (precision, recall, f1).

    We compute the predictions ONCE and then only re-rank for each N, which
    is much faster than re-predicting for every N.
    """
    results = {}

    for n in n_values:
        precisions, recalls = precision_recall_at_n(
            predictions,
            n=n,
            threshold=threshold
        )

        # Average over all users (each user counts the same).
        pre = sum(precisions.values()) / len(precisions) if precisions else 0.0
        rec = sum(recalls.values()) / len(recalls) if recalls else 0.0

        # F1 = harmonic mean of precision and recall.
        f1 = 2 * pre * rec / (pre + rec) if (pre + rec) > 0 else 0.0

        results[n] = (pre, rec, f1)

    return results


def evaluate_top_n(trainset, testset, algo, label,
                   n_values=range(10, 101), threshold=4):
    """
    Fit a model, predict the test set once, then compute Precision/Recall/F1
    for every N in n_values. Prints a table and returns the results dict.
    """
    algo.fit(trainset)                  # train the model
    predictions = algo.test(testset)    # predict every rating in the test set
    results = top_n_metrics(predictions, n_values, threshold)

    print(f"\n=== {label} ===")
    print(f"{'N':>4} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 44)
    for n, (pre, rec, f1) in results.items():
        print(f"{n:>4} | {pre:>10.6f} | {rec:>10.6f} | {f1:>10.6f}")

    return results


def plot_top_n_results(results_knn, results_svd, title, filename):
    """
    Draw one figure with 3 subplots (Precision, Recall, F1) versus N.
    In each subplot we compare the K-NN model with the SVD model.
    The figure is saved to `filename` (png)
    """
    # The N values are the keys of the results dict (insertion order = 10..100).
    ns = list(results_knn.keys())

    # (metric name, position in the (precision, recall, f1) tuple)
    metrics = [("Precision", 0), ("Recall", 1), ("F1", 2)]

    # 1 row x 3 columns of subplots.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, (name, idx) in zip(axes, metrics):
        # For each metric, collect its value for every N and plot a line.
        ax.plot(
            ns,
            [results_knn[n][idx] for n in ns],
            label="K-NN",
            marker="o",
            markersize=3
        )
        ax.plot(
            ns,
            [results_svd[n][idx] for n in ns],
            label="SVD",
            marker="s",
            markersize=3
        )

        ax.set_title(name)
        ax.set_xlabel("N (number of recommendations)")
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)   # light grid to read values more easily
        ax.legend()

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)                 # free memory
    print(f"Saved graph: {filename}")

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

# PART 3 -----------------------------------------------------------------
if todo_3:
    N_values = list(range(10, 101))

    # -----------------------------------------------------------
    # 25% of the ratings withheld  (same split as tasks 1a and 2)
    # -----------------------------------------------------------
    trainset3_25, testset3_25 = train_test_split(
        data,
        test_size=0.25,
        random_state=22
    )

    # User-based K-NN with the best K from task 1a (K=75).
    knn_25 = KNNWithMeans(
        k=75,
        sim_options={"name": "pearson", "user_based": True}  # neighbours are users
    )

    # SVD on the same split; random_state=3 is required by the assignment.
    svd_25 = SVD(random_state=3)

    results_knn_25 = evaluate_top_n(
        trainset3_25,
        testset3_25,
        knn_25,
        "KNN (K=75), 25% missing",
        N_values
    )

    results_svd_25 = evaluate_top_n(
        trainset3_25,
        testset3_25,
        svd_25,
        "SVD, 25% missing",
        N_values
    )

    # -----------------------------------------------------------
    # 75% of the ratings withheld  (same split as tasks 1b and 2)
    # -----------------------------------------------------------
    trainset3_75, testset3_75 = train_test_split(
        data,
        test_size=0.75,
        random_state=22
    )

    # User-based K-NN with the best K from task 1b (K=61).
    knn_75 = KNNWithMeans(
        k=61,
        sim_options={"name": "pearson", "user_based": True}
    )

    svd_75 = SVD(random_state=3)

    results_knn_75 = evaluate_top_n(
        trainset3_75,
        testset3_75,
        knn_75,
        "KNN (K=61), 75% missing",
        N_values
    )

    results_svd_75 = evaluate_top_n(
        trainset3_75,
        testset3_75,
        svd_75,
        "SVD, 75% missing",
        N_values
    )

    # Graphs -------------------------------------------------
    # One figure per scenario (25% / 75% missing ratings).
    # Each figure has Precision, Recall and F1 vs N, K-NN vs SVD.
    plot_top_n_results(
        results_knn_25,
        results_svd_25,
        "Top-N recommendations, 25% missing ratings",
        "topn_25.png"
    )

    plot_top_n_results(
        results_knn_75,
        results_svd_75,
        "Top-N recommendations, 75% missing ratings",
        "topn_75.png"
    )

