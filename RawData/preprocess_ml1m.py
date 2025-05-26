import os
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import json


def load_data(data_dir):
    ratings = pd.read_csv(
        os.path.join(data_dir, "ratings.dat"),
        sep="::",
        engine="python",
        names=["UserID", "MovieID", "Rating", "Timestamp"],
        encoding="latin-1",
    )
    users = pd.read_csv(
        os.path.join(data_dir, "users.dat"),
        sep="::",
        engine="python",
        names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
        encoding="latin-1",
    )
    movies = pd.read_csv(
        os.path.join(data_dir, "movies.dat"),
        sep="::",
        engine="python",
        names=["MovieID", "Title", "Genres"],
        encoding="latin-1",
    )
    return ratings, users, movies


def build_genre_map(movies, out_dir):
    genre_set = set()
    for genres in movies["Genres"]:
        genre_set.update(genres.split("|"))
    genre_list = sorted(list(genre_set))
    genre2idx = {g: i for i, g in enumerate(genre_list)}
    with open(os.path.join(out_dir, "genre_map.json"), "w") as f:
        json.dump(genre2idx, f)
    return genre2idx


def encode_genre(genres, genre2idx):
    arr = np.zeros(len(genre2idx), dtype=np.float32)
    for g in genres.split("|"):
        if g in genre2idx:
            arr[genre2idx[g]] = 1.0
    return arr


def build_title_embeddings(movies):
    print("Encoding movie titles...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    titles = movies["Title"].tolist()
    title_embs = model.encode(
        titles, show_progress_bar=True, batch_size=512, desc="Encoding movie titles"
    )
    return title_embs


def build_item_features(movies, title_embs, genre2idx, out_dir):
    itemid2idx = {mid: i for i, mid in enumerate(movies["MovieID"])}
    item_features = []
    for i, row in movies.iterrows():
        genre_vec = encode_genre(row["Genres"], genre2idx)
        emb = title_embs[i]
        feat = np.concatenate([emb, genre_vec])
        item_features.append(feat)
    item_features = np.stack(item_features)
    np.save(os.path.join(out_dir, "item_features.npy"), item_features)
    with open(os.path.join(out_dir, "itemid2idx.pkl"), "wb") as f:
        pickle.dump(itemid2idx, f)
    np.save(os.path.join(out_dir, "item2title_emb.npy"), title_embs)
    return itemid2idx, item_features


def build_user_features(users, out_dir):
    gender_map = {"M": 0, "F": 1}
    age_list = sorted(users["Age"].unique())
    age2idx = {a: i for i, a in enumerate(age_list)}
    occ_list = sorted(users["Occupation"].unique())
    occ2idx = {o: i for i, o in enumerate(occ_list)}
    user_features = []
    for _, row in users.iterrows():
        gender = gender_map.get(row["Gender"], 0)
        age = age2idx[row["Age"]]
        occ = occ2idx[row["Occupation"]]
        zip_hash = hash(row["Zip-code"]) % 10000 / 10000.0
        user_features.append([gender, age, occ, zip_hash])
    user_features = np.stack(user_features)
    np.save(os.path.join(out_dir, "user_features.npy"), user_features)
    return user_features


def build_user_sequences_and_splits(ratings, out_dir):
    user_train = {}
    user_neg = {}
    user_train_split = {}
    user_valid = {}
    user_test = {}
    for uid, group in tqdm(ratings.groupby("UserID"), desc="Building user sequences"):
        pos = group[group["Rating"] > 3].sort_values("Timestamp")["MovieID"].tolist()
        neg = group[group["Rating"] <= 3].sort_values("Timestamp")["MovieID"].tolist()
        # Only keep users with at least 3 positive interactions
        if len(pos) < 3:
            continue
        user_train[uid] = pos
        if len(neg) > 0:
            user_neg[uid] = neg
        # Train/val/test split
        user_train_split[uid] = pos[:-2]
        user_valid[uid] = pos[-2]
        user_test[uid] = pos[-1]
    with open(os.path.join(out_dir, "user_train.pkl"), "wb") as f:
        pickle.dump(user_train, f)
    with open(os.path.join(out_dir, "user_neg.pkl"), "wb") as f:
        pickle.dump(user_neg, f)
    with open(os.path.join(out_dir, "user_train_split.pkl"), "wb") as f:
        pickle.dump(user_train_split, f)
    with open(os.path.join(out_dir, "user_valid.pkl"), "wb") as f:
        pickle.dump(user_valid, f)
    with open(os.path.join(out_dir, "user_test.pkl"), "wb") as f:
        pickle.dump(user_test, f)
    return user_train, user_neg, user_train_split, user_valid, user_test


def timestamp_features(ts, min_ts, max_ts):
    import datetime

    dt = datetime.datetime.fromtimestamp(ts)
    hour = dt.hour / 23.0
    day = dt.weekday() / 6.0
    norm_ts = (ts - min_ts) / (max_ts - min_ts)
    return np.array([hour, day, norm_ts], dtype=np.float32)


def build_context_dict(ratings, movies, title_embs, genre2idx, itemid2idx, out_dir):
    min_ts = ratings["Timestamp"].min()
    max_ts = ratings["Timestamp"].max()
    cxtdict = {}
    for _, row in tqdm(
        ratings.iterrows(), total=len(ratings), desc="Building context dict"
    ):
        uid = row["UserID"]
        iid = row["MovieID"]
        ts = row["Timestamp"]
        rating = row["Rating"]
        if iid in itemid2idx:
            title_emb = title_embs[itemid2idx[iid]]
            genre_vec = encode_genre(movies.loc[itemid2idx[iid], "Genres"], genre2idx)
            ts_feat = timestamp_features(ts, min_ts, max_ts)
            cxt = np.concatenate([ts_feat, title_emb, genre_vec, [rating]])
            cxtdict[(uid, iid)] = cxt
    cxtsize = len(next(iter(cxtdict.values())))
    with open(os.path.join(out_dir, "cxtdict.pkl"), "wb") as f:
        pickle.dump(cxtdict, f)
    with open(os.path.join(out_dir, "cxtsize.txt"), "w") as f:
        f.write(str(cxtsize))
    return cxtdict, cxtsize


def build_ratings_matrix(ratings, users, movies, out_dir):
    user_ids_sorted = sorted(users["UserID"].unique())
    item_ids_sorted = sorted(movies["MovieID"].unique())
    userid2idx = {uid: i for i, uid in enumerate(user_ids_sorted)}
    itemid2col = {iid: i for i, iid in enumerate(item_ids_sorted)}
    ratings_matrix = np.zeros(
        (len(user_ids_sorted), len(item_ids_sorted)), dtype=np.float32
    )
    for _, row in ratings.iterrows():
        uid = row["UserID"]
        iid = row["MovieID"]
        rating = row["Rating"]
        if uid in userid2idx and iid in itemid2col:
            ratings_matrix[userid2idx[uid], itemid2col[iid]] = rating
    np.save(os.path.join(out_dir, "ratings_matrix.npy"), ratings_matrix)
    with open(os.path.join(out_dir, "userid2idx.pkl"), "wb") as f:
        pickle.dump(userid2idx, f)
    # Optionally save itemid2col for reference
    with open(os.path.join(out_dir, "itemid2col.pkl"), "wb") as f:
        pickle.dump(itemid2col, f)
    return ratings_matrix, userid2idx, itemid2col


def main():
    DATA_DIR = "./RawData/ml-1m"
    OUT_DIR = "Data/movielens_preprocessed"
    os.makedirs(OUT_DIR, exist_ok=True)

    ratings, users, movies = load_data(DATA_DIR)
    genre2idx = build_genre_map(movies, OUT_DIR)
    title_embs = build_title_embeddings(movies)
    itemid2idx, item_features = build_item_features(
        movies, title_embs, genre2idx, OUT_DIR
    )
    user_features = build_user_features(users, OUT_DIR)
    user_train, user_neg, user_train_split, user_valid, user_test = (
        build_user_sequences_and_splits(ratings, OUT_DIR)
    )
    cxtdict, cxtsize = build_context_dict(
        ratings, movies, title_embs, genre2idx, itemid2idx, OUT_DIR
    )
    ratings_matrix, userid2idx, itemid2col = build_ratings_matrix(
        ratings, users, movies, OUT_DIR
    )
    print("Preprocessing complete. Files saved in", OUT_DIR)


if __name__ == "__main__":
    main()
