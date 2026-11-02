# Adaptive Systems – Assignment 2 (MovieLens + Surprise)

Implementation of the assignment using the **Surprise** library for collaborative
filtering on the MovieLens datasets.

All the code is in [`main.py`](main.py). Each task is guarded by a `todo_*`
flag at the top of the file, so you can run only the parts you need.

---

## 1. What does "X% of missing ratings" mean?

The dataset **only contains observed ratings** (there are no empty/NaN values in
`u.data`). "Missing ratings" therefore does **not** describe the file — it means
we *artificially hide* a percentage of the known ratings from the model:

- The hidden ratings are removed from the training set → the model never sees them.
- Those same hidden ratings form the **test set**, used to evaluate the model.

So:

| Statement | Training set | Test set | `train_test_split` |
|---|---|---|---|
| 25% of missing ratings | 75% of ratings | 25% of ratings | `test_size=0.25` |
| 75% of missing ratings | 25% of ratings | 75% of ratings | `test_size=0.75` |

"Missing" is just another way of saying "withheld for evaluation". The 75% case
is the *sparsity problem*: far less training data is available, so predictions
get worse.

All splits use `random_state=22`, as required. The SVD model uses
`random_state=3`, also as required. These two seeds are **not** interchangeable:
`22` is for the train/test split, `3` is for constructing `SVD(...)`.

---

## 2. Project structure

```
adaptiveSys-A2/
├── main.py                 # all tasks (1a, 1b, 2, 3, 4)
├── README.md
├── u_data.csv              # copy of ml-100k/u.data (ratings)
├── ml-100k/                # MovieLens 100k dataset
│   ├── u.data              # user_id | item_id | rating | timestamp
│   ├── u.item              # movie metadata (NOT needed here)
│   └── ...
├── ml-1m/                  # MovieLens 1M dataset (task 4 only)
│   └── ratings.dat         # user_id::item_id::rating::timestamp
├── topn_25.png             # generated graph: 25% missing
└── topn_75.png             # generated graph: 75% missing
```

**Which data file is used?** `u.data` (loaded from `u_data.csv`), because it
contains the ratings. `u.item` only holds movie metadata (title, genres) and is
not needed for these tasks. The pre-split `u*.base` / `u*.test` files are 80/20
splits, which do **not** match the required 25/75 ratios, so they are not used
either.

---

## 3. Requirements

```bash
pip install "scikit-surprise==1.1.5" pandas numpy matplotlib
```

The code was run with Surprise 1.1.5, pandas 3.1.0, numpy 2.5.4 and
matplotlib 3.11.2. MovieLens 100k is already in the repository. MovieLens 1M
(task 4) is downloaded to `ml-1m/` with:

```bash
curl -sSLo ml-1m.zip https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
```

---

## 4. How to run

Set the flags at the top of `main.py`:

```python
todo_1a = False
todo_1b = False
todo_2  = False
todo_3  = False   # also regenerates the two graphs
todo_4  = True
```

then:

```bash
python main.py
```

Notes:

- Task 1 (`todo_1a` / `todo_1b`) sweeps `K = 1..100` and is the slowest part.
- Task 3 computes predictions once per model and re-ranks them for every
  `N`, then saves `topn_25.png` and `topn_75.png`.
- Task 4 needs `ml-1m/ratings.dat`.

---

## 5. Methodology

- **Ratings** are loaded from the long-format files
  (`user_id, item_id, rating`). A `Reader(rating_scale=(1, 5))` tells Surprise
  the valid rating range so predictions can be clipped to `[1, 5]`.
- **User-based K-NN** = `KNNWithMeans` with
  `sim_options={"name": "pearson", "user_based": True}`: neighbours are users
  and similarity is the Pearson correlation between users' rating vectors.
- **SVD** = Surprise's Funk-SVD implementation (`SVD(random_state=3)`).
- **MAE** = mean absolute error between predicted and true ratings.
- **Top-N relevance**: an item is relevant for a user if its *true* rating is
  **4 or 5** (`threshold=4`).
  - `Precision@N = hits / min(N, number of test items for the user)`
  - `Recall@N = hits / number of relevant test items for the user`
  - `F1@N = 2 * Precision * Recall / (Precision + Recall)`
  - Only items in the test set are ranked (no ground truth exists for unseen
    items). Predictions are computed once and re-ranked for each `N`.

---

## 6. Results

### Task 1 – Best K for user-based K-NN

| Scenario | Best K | MAE |
|---|---|---|
| 1a: 25% missing | **75** | **0.7446623579** |
| 1b: 75% missing | **61** | **0.8125340341** |

MAE is higher with 75% missing because the model has much less training data
(the sparsity problem).

### Task 2 – SVD vs K-NN (MAE)

| Missing | K-NN | SVD |
|---|---|---|
| 25% | 0.7447 | **0.7388** |
| 75% | 0.8125 | **0.7736** |

SVD is better in both cases, and the improvement is much larger with 75%
missing. K-NN relies on user-user correlations, which become noisy when data is
scarce; SVD learns latent factors that generalise better.

### Task 3 – Top-N (Precision / Recall / F1)

Selected values (full tables are printed by `main.py`; N goes from 10 to 100):

| Model | N | Precision | Recall | F1 |
|---|---|---|---|---|
| K-NN K=75, 25% | 10 | 0.6922 | 0.6657 | 0.6787 |
| K-NN K=75, 25% | 40 | 0.6064 | 0.9451 | **0.7388** |
| SVD, 25% | 10 | 0.6903 | 0.6642 | 0.6770 |
| SVD, 25% | 40 | 0.6057 | 0.9443 | **0.7380** |
| K-NN K=61, 75% | 10 | 0.7033 | 0.3049 | 0.4254 |
| K-NN K=61, 75% | 50 | 0.6367 | 0.7895 | 0.7050 |
| SVD, 75% | 10 | 0.7487 | 0.3264 | **0.4546** |
| SVD, 75% | 50 | 0.6426 | 0.7940 | **0.7103** |

Generated graphs: `topn_25.png` and `topn_75.png`.

**Interpretation**

- Precision decreases with N and recall increases with N (we recommend more
  items, so we hit more relevant ones but also more irrelevant ones). F1 is the
  compromise and peaks around N ≈ 40.
- At 25% missing, K-NN and SVD are almost identical. Both are trained on the
  same split and rank the same test items, and their predicted ratings differ
  very little (MAE gap ≈ 0.006). With ~25 test items per user, the candidate set
  is small, so both models rank it similarly.
- At 75% missing, SVD is clearly better, especially for small N: precision at
  N=10 is 0.749 vs 0.703 and F1 is 0.455 vs 0.425. This confirms that factoring
  helps when the user-item matrix is very sparse.
- Recall is lower at 75% for small N because each user still has many relevant
  items, but the models have much less training data to identify them.

### Task 4 – MovieLens 1M (N=10, 25% missing)

| Model / dataset | Precision | Recall | F1 |
|---|---|---|---|
| ML-100k KNN | 0.6922 | 0.6657 | 0.6787 |
| ML-1M KNN | 0.7580 | 0.5718 | 0.6519 |
| ML-100k SVD | 0.6903 | 0.6642 | 0.6770 |
| ML-1M SVD | 0.7690 | 0.5763 | 0.6589 |

**Why the differences make sense**

- **Higher precision on 1M.** Items have far more ratings (~270 on average vs
  ~60 on 100k) and users have longer histories (~166 vs ~106 ratings). Better
  statistics per item and per user make the top-10 list more reliably relevant.
- **Lower recall on 1M.** Recall is `hits / all relevant items for the user`.
  In 1M each user has many more relevant (4–5★) items in the test set, while N
  is still only 10, so a fixed top-10 covers a smaller fraction of them.
- **Slightly lower F1 on 1M.** The gain in precision does not compensate for the
  loss in recall (F1 is the harmonic mean).
- Although the 1M user-item matrix is *globally* sparser (bigger catalogue:
  6040 × 3706, ≈4.5% filled vs ≈6.3% for 100k), the absolute number of ratings
  per user and per item is much larger, which is what drives these results.
- SVD still beats K-NN on 1M, consistent with the sparsity discussion.

**Assumption:** for Task 4 the best K from Task 1a (`K=75`) was reused instead
of re-tuning K on the 1M dataset. Retuning would require sweeping 100 K values
over ~1M ratings, which is prohibitively slow and is not explicitly required by
the statement. The optimal K for 1M could differ.

---

## 7. Reminder of the required seeds

```python
train_test_split(data, test_size=..., random_state=22)  # always 22
SVD(random_state=3)                                     # always 3
```
