from collections import defaultdict
from pprint import pprint

import pandas as pd
from surprise import KNNWithMeans, Reader, Dataset
from surprise.accuracy import mae
from sklearn.model_selection import train_test_split


# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')

def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """
    Return precision and recall at n metrics for each user.
    Precision : true positives / (true positives + false positives)
    Recall = true positives / (true positives + false negatives)
    Threshold : A movie is relevant for the user if itf rating is >= threshold
    """

    user_est_true = defaultdict(list)

    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum(
            true_r >= threshold
            for _, true_r in user_ratings
        )

        n_rel_and_rec = sum(
            true_r >= threshold
            for _, true_r in user_ratings[:n]
        )

        precisions[uid] = n_rel_and_rec / min(n, len(user_ratings))
        recalls[uid] = (
            n_rel_and_rec / n_rel if n_rel != 0 else 0
        )

    return precisions, recalls


def load_csv():
    """Load ratings from a matrix CSV and return a long-format DataFrame."""

    csv_file = pd.read_csv(
        "data.csv",
        delimiter=";",
        dtype=str
    )

    # First column = item IDs
    item_ids = csv_file.iloc[:, 0]

    # Remaining column headers = user IDs
    ratings_matrix = csv_file.iloc[:, 1:]

    # Add item IDs to the ratings matrix
    df = ratings_matrix.copy()
    df.insert(0, "itemID", item_ids)

    # Convert matrix format to long format
    df = df.melt(
        id_vars="itemID",
        var_name="userID",
        value_name="rating"
    )

    # Remove missing ratings
    df = df.dropna(subset=["rating"])

    # Convert ratings to numeric; keep IDs as strings
    df["rating"] = pd.to_numeric(df["rating"])

    # Ensure the column order required by Surprise
    return df[["userID", "itemID", "rating"]]


# ---------------------------------------------------------
# Load ratings
# ---------------------------------------------------------

df = load_csv()

print("Total available ratings:", len(df))

# ---------------------------------------------------------
# Random train/test split: 75% training, 25% test
# ---------------------------------------------------------

train_df, test_df = train_test_split(
    df,
    test_size=0.25,
    random_state=20
)

# Convert training data to Surprise format
reader = Reader(rating_scale=(1, 5))

train_data = Dataset.load_from_df(
    train_df[["userID", "itemID", "rating"]],
    reader
)

trainset25 = train_data.build_full_trainset()

# Keep original string IDs in the test set
testset25 = list(
    test_df[["userID", "itemID", "rating"]]
    .itertuples(index=False, name=None)
)

print("Training ratings:", len(train_df))
print("Test ratings:", len(test_df))

# ---------------------------------------------------------
# User-based KNN
# ---------------------------------------------------------

sim_options_KNN = {
    "name": "pearson",
    "user_based": True
}

k = 3

algo = KNNWithMeans(
    k,
    sim_options=sim_options_KNN,
    verbose=False
)

algo.fit(trainset25)

# ---------------------------------------------------------
# Predict ratings in the test set
# ---------------------------------------------------------

predictions25KNN = algo.test(testset25)

pprint(predictions25KNN)

# Display predictions for a specific user
target_user = "Batman"  # Replace with an actual user ID from your CSV

for uid, iid, real, est, _ in predictions25KNN:
    if uid == target_user:
        print(f"User: {uid}, Item: {iid}, Actual: {real}, Estimated: {est}")

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

mae(predictions25KNN)

precisions, recalls = precision_recall_at_n(
    predictions25KNN,
    n=5,
    threshold=4
)

pre = sum(precisions.values()) / len(precisions) if precisions else 0
recall = sum(recalls.values()) / len(recalls) if recalls else 0

print("Precision:", pre)
print("Recall:", recall)

f1 = (
    2 * pre * recall / (pre + recall)
    if pre + recall > 0
    else 0
)

print("F1:", f1)
