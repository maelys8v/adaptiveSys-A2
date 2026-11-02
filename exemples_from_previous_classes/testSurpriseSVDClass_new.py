from collections import defaultdict
from pprint import pprint

import pandas as pd
from surprise import Reader, SVD, Dataset
from surprise.accuracy import mae
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
# data = Dataset.load_builtin('ml-100k')


def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """Calculate standard Precision@N and Recall@N for each user."""

    user_est_true = defaultdict(list)

    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = {}
    recalls = {}

    for uid, user_ratings in user_est_true.items():
        # Sort by estimated rating, highest first
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items in the test set
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)

        # Number of relevant items among the top N recommendations
        n_rel_and_rec = sum(
            true_r >= threshold
            for _, true_r in user_ratings[:n]
        )

        # Standard Precision@N: relevant recommendations / N
        precisions[uid] = n_rel_and_rec / min(n, len(user_ratings))

        # Recall@N: relevant recommendations / relevant test items
        recalls[uid] = (
            n_rel_and_rec / n_rel if n_rel > 0 else 0
        )

    return precisions, recalls


def load_csv():
    """Load a user-item rating matrix from data.csv."""

    # First column: item IDs
    # Remaining column headers: user IDs
    csv_file = pd.read_csv(
        "data.csv",
        delimiter=";",
        dtype=str
    )

    item_ids = csv_file.iloc[:, 0]
    ratings_matrix = csv_file.iloc[:, 1:].copy()

    # Associate each row with its item ID
    ratings_matrix.insert(0, "itemID", item_ids)

    # Convert the matrix into long format:
    # one row per user-item-rating combination
    df = ratings_matrix.melt(
        id_vars="itemID",
        var_name="userID",
        value_name="rating"
    )

    # Remove missing ratings
    df = df.dropna(subset=["rating"])

    # Convert ratings to numeric values
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])

    # Prepare the data for Surprise
    reader = Reader(rating_scale=(1, 5))

    return Dataset.load_from_df(
        df[["userID", "itemID", "rating"]],
        reader
    )


# Load ratings from CSV
data = load_csv()

# Split the observed ratings into training and test sets
trainset25, testset25 = train_test_split(
    data,
    test_size=0.25,
    random_state=22
)

# Train the SVD model
algo = SVD(random_state=3)
algo.fit(trainset25)

# Predict ratings for the test set
predictions25SVD = algo.test(testset25)

pprint(predictions25SVD)

# Display predictions for a specific user ID
target_user = "Batman"

for uid, iid, real, est, _ in predictions25SVD:
    if uid == target_user:
        print(f"User: {uid}, Item: {iid}, Real: {real}, Estimated: {est}")

# Calculate MAE
mae(predictions25SVD)

# Calculate Precision@5 and Recall@5
precisions, recalls = precision_recall_at_n(
    predictions25SVD,
    n=5,
    threshold=4
)

# Average across users
pre = sum(precisions.values()) / len(precisions) if precisions else 0
recall = sum(recalls.values()) / len(recalls) if recalls else 0

f1 = (
    2 * pre * recall / (pre + recall)
    if pre + recall > 0
    else 0
)

print("Precision@5:", pre)
print("Recall@5:", recall)
print("F1:", f1)
