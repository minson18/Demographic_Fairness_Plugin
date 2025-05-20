import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Paths
DATA_DIR = "./RawData/ml-1m"
OUT_DIR = "Data/movielens_preprocessed"
os.makedirs(OUT_DIR, exist_ok=True)

# 1. Load ratings
ratings = pd.read_csv(
    os.path.join(DATA_DIR, "ratings.dat"),
    sep="::",
    engine="python",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
    encoding="latin-1",
)

# 2. Load users
users = pd.read_csv(
    os.path.join(DATA_DIR, "users.dat"),
    sep="::",
    engine="python",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
    encoding="latin-1",
)

# 3. Load movies
movies = pd.read_csv(
    os.path.join(DATA_DIR, "movies.dat"),
    sep="::",
    engine="python",
    names=["MovieID", "Title", "Genres"],
    encoding="latin-1",
)

# 4. Build genre map and multi-hot encoding
genre_set = set()
for genres in movies["Genres"]:
    genre_set.update(genres.split("|"))
genre_list = sorted(list(genre_set))
genre2idx = {g: i for i, g in enumerate(genre_list)}
with open(os.path.join(OUT_DIR, "genre_map.json"), "w") as f:
    import json

    json.dump(genre2idx, f)


def encode_genre(genres):
    arr = np.zeros(len(genre2idx), dtype=np.float32)
    for g in genres.split("|"):
        if g in genre2idx:
            arr[genre2idx[g]] = 1.0
    return arr


# 5. Title embedding
print("Encoding movie titles...")
model = SentenceTransformer("all-MiniLM-L6-v2")
titles = movies["Title"].tolist()
title_embs = model.encode(
    titles, show_progress_bar=True, batch_size=128, desc="Encoding movie titles"
)

# 6. Build item features (title embedding + genre)
itemid2idx = {mid: i for i, mid in enumerate(movies["MovieID"])}
item_features = []
for i, row in movies.iterrows():
    genre_vec = encode_genre(row["Genres"])
    emb = title_embs[i]
    feat = np.concatenate([emb, genre_vec])
    item_features.append(feat)
item_features = np.stack(item_features)
np.save(os.path.join(OUT_DIR, "item_features.npy"), item_features)
with open(os.path.join(OUT_DIR, "itemid2idx.pkl"), "wb") as f:
    pickle.dump(itemid2idx, f)

# 7. Build user features (Gender, Age, Occupation, Zip-code)
gender_map = {"M": 0, "F": 1}
age_list = sorted(users["Age"].unique())
age2idx = {a: i for i, a in enumerate(age_list)}
occ_list = sorted(users["Occupation"].unique())
occ2idx = {o: i for i, o in enumerate(occ_list)}

user_features = []
user_ids = users["UserID"].tolist()
for _, row in users.iterrows():
    gender = gender_map.get(row["Gender"], 0)
    age = age2idx[row["Age"]]
    occ = occ2idx[row["Occupation"]]
    # Zip-code as int hash (or ignore for now)
    zip_hash = hash(row["Zip-code"]) % 10000 / 10000.0
    user_features.append([gender, age, occ, zip_hash])
user_features = np.stack(user_features)
np.save(os.path.join(OUT_DIR, "user_features.npy"), user_features)

# 8. Build user_train (user -> [item sequence])
user_train = {}
user_neg = {}
for uid, group in tqdm(ratings.groupby("UserID"), desc="Building user sequences"):
    pos = group[group["Rating"] > 3].sort_values("Timestamp")["MovieID"].tolist()
    neg = group[group["Rating"] <= 3].sort_values("Timestamp")["MovieID"].tolist()
    if len(pos) > 1:
        user_train[uid] = pos
    if len(neg) > 0:
        user_neg[uid] = neg
with open(os.path.join(OUT_DIR, "user_train.pkl"), "wb") as f:
    pickle.dump(user_train, f)
with open(os.path.join(OUT_DIR, "user_neg.pkl"), "wb") as f:
    pickle.dump(user_neg, f)


# 9. Build context dict: (user, item) -> [timestamp features + title emb + genre]
def timestamp_features(ts):
    # Hour of day, day of week, normalized timestamp
    import datetime

    dt = datetime.datetime.fromtimestamp(ts)
    hour = dt.hour / 23.0
    day = dt.weekday() / 6.0
    norm_ts = (ts - ratings["Timestamp"].min()) / (
        ratings["Timestamp"].max() - ratings["Timestamp"].min()
    )
    return np.array([hour, day, norm_ts], dtype=np.float32)


cxtdict = {}
cxtsize = 3 + title_embs.shape[1] + len(genre2idx)
for _, row in tqdm(
    ratings.iterrows(), total=len(ratings), desc="Building context dict"
):
    uid = row["UserID"]
    iid = row["MovieID"]
    ts = row["Timestamp"]
    if iid in itemid2idx:
        title_emb = title_embs[itemid2idx[iid]]
        genre_vec = encode_genre(movies.loc[itemid2idx[iid], "Genres"])
        ts_feat = timestamp_features(ts)
        cxt = np.concatenate([ts_feat, title_emb, genre_vec])
        cxtdict[(uid, iid)] = cxt
cxtsize = len(next(iter(cxtdict.values())))
with open(os.path.join(OUT_DIR, "cxtsize.txt"), "w") as f:
    f.write(str(cxtsize))

# 10. Save item2title_emb for fast lookup
np.save(os.path.join(OUT_DIR, "item2title_emb.npy"), title_embs)

print("Preprocessing complete. Files saved in", OUT_DIR)
