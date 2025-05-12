import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
import os


def load_data(filename):
    try:
        with open(filename, "rb") as f:
            x = pickle.load(f)
    except Exception:
        x = []
    return x


def save_data(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def data_partition(fname):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(f"./Data/{fname}.txt", "r") as f:
        for line in f:
            u, i = line.rstrip().split(" ")
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            User[u].append(i)
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = [User[user][-2]]
            user_test[user] = [User[user][-1]]
    return [user_train, user_valid, user_test, usernum, itemnum]


def PreprocessData(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine="python")
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    df["ts"] = pd.to_datetime(df["st"], unit="s")
    df = df.sort_values(by=["ts"])
    df["year"], df["month"], df["day"], df["dayofweek"], df["dayofyear"], df["week"] = (
        zip(
            *df["ts"].map(
                lambda x: [
                    x.year,
                    x.month,
                    x.day,
                    x.dayofweek,
                    x.dayofyear,
                    x.isocalendar()[1],
                ]
            )
        )
    )
    df["year"] -= df["year"].min()
    df["year"] /= df["year"].max() if df["year"].max() > 0 else 1
    df["month"] /= 12
    df["day"] /= 31
    df["dayofweek"] /= 7
    df["dayofyear"] /= 365
    df["week"] /= 4
    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows():
        userid = int(row["user"])
        itemid = int(row["item"])
        if userid in UsersDict:
            UsersDict[userid].append(itemid)
        else:
            UsersDict[userid] = [itemid]
        year = row["year"]
        month = row["month"]
        day = row["day"]
        dayofweek = row["dayofweek"]
        dayofyear = row["dayofyear"]
        week = row["week"]
        DATEINFO[(userid, itemid)] = [year, month, day, dayofweek, dayofyear, week]
    return df, DATEINFO


def get_ItemDataBeauty(itemnum):
    ItemFeatures = load_data("./Data/Beauty_feat_cat.dat")
    ItemFeatures = np.vstack(
        (np.zeros((1, ItemFeatures.shape[1]), dtype=ItemFeatures.dtype), ItemFeatures)
    )
    return ItemFeatures


def get_UserDataBeauty(usernum):
    UserFeatures = np.identity(usernum, dtype=np.int8)
    UserFeatures = np.vstack((np.zeros((1, usernum), dtype=np.int8), UserFeatures))
    return UserFeatures


def PreprocessData_Beauty(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine="python")
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(by=["ts"])
    df["year"], df["month"], df["day"], df["dayofweek"], df["dayofyear"], df["week"] = (
        zip(
            *df["ts"].map(
                lambda x: [
                    x.year,
                    x.month,
                    x.day,
                    x.dayofweek,
                    x.dayofyear,
                    x.isocalendar()[1],
                ]
            )
        )
    )
    df["year"] -= df["year"].min()
    df["year"] /= df["year"].max() if df["year"].max() > 0 else 1
    df["month"] /= 12
    df["day"] /= 31
    df["dayofweek"] /= 7
    df["dayofyear"] /= 365
    df["week"] /= 4
    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows():
        userid = int(row["user"])
        itemid = int(row["item"])
        year = row["year"]
        month = row["month"]
        day = row["day"]
        dayofweek = row["dayofweek"]
        dayofyear = row["dayofyear"]
        week = row["week"]
        DATEINFO[(userid, itemid)] = [year, month, day, dayofweek, dayofyear, week]
    return df, DATEINFO


def get_ItemDataMen(itemnum):
    ItemFeatures = load_data("./Data/Men_imgs.dat")
    ItemFeatures = np.vstack(
        (np.zeros((1, ItemFeatures.shape[1]), dtype=ItemFeatures.dtype), ItemFeatures)
    )
    return ItemFeatures


def get_UserDataMen(usernum):
    UserFeatures = np.identity(usernum, dtype=np.int8)
    UserFeatures = np.vstack((np.zeros((1, usernum), dtype=np.int8), UserFeatures))
    return UserFeatures


def PreprocessData_Men(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine="python")
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(by=["ts"])
    df["year"], df["month"], df["day"], df["dayofweek"], df["dayofyear"], df["week"] = (
        zip(
            *df["ts"].map(
                lambda x: [
                    x.year,
                    x.month,
                    x.day,
                    x.dayofweek,
                    x.dayofyear,
                    x.isocalendar()[1],
                ]
            )
        )
    )
    df["year"] -= df["year"].min()
    df["year"] /= df["year"].max() if df["year"].max() > 0 else 1
    df["month"] /= 12
    df["day"] /= 31
    df["dayofweek"] /= 7
    df["dayofyear"] /= 365
    df["week"] /= 4
    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows():
        userid = int(row["user"])
        itemid = int(row["item"])
        year = row["year"]
        month = row["month"]
        day = row["day"]
        dayofweek = row["dayofweek"]
        dayofyear = row["dayofyear"]
        week = row["week"]
        DATEINFO[(userid, itemid)] = [year, month, day, dayofweek, dayofyear, week]
    return df, DATEINFO


def PreprocessData_Fashion(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine="python")
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(by=["ts"])
    df["year"], df["month"], df["day"], df["dayofweek"], df["dayofyear"], df["week"] = (
        zip(
            *df["ts"].map(
                lambda x: [
                    x.year,
                    x.month,
                    x.day,
                    x.dayofweek,
                    x.dayofyear,
                    x.isocalendar()[1],
                ]
            )
        )
    )
    df["year"] -= df["year"].min()
    df["year"] /= df["year"].max() if df["year"].max() > 0 else 1
    df["month"] /= 12
    df["day"] /= 31
    df["dayofweek"] /= 7
    df["dayofyear"] /= 365
    df["week"] /= 4
    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows():
        userid = int(row["user"])
        itemid = int(row["item"])
        year = row["year"]
        month = row["month"]
        day = row["day"]
        dayofweek = row["dayofweek"]
        dayofyear = row["dayofyear"]
        week = row["week"]
        DATEINFO[(userid, itemid)] = [year, month, day, dayofweek, dayofyear, week]
    return df, DATEINFO


def get_ItemDataFashion(itemnum):
    ItemFeatures = load_data("./Data/Fashion_imgs.dat")
    ItemFeatures = np.vstack(
        (np.zeros((1, ItemFeatures.shape[1]), dtype=ItemFeatures.dtype), ItemFeatures)
    )
    return ItemFeatures


def get_UserDataFashion(usernum):
    UserFeatures = np.identity(usernum, dtype=np.int8)
    UserFeatures = np.vstack((np.zeros((1, usernum), dtype=np.int8), UserFeatures))
    return UserFeatures


def PreprocessData_Games(filname, DatasetName, sep="\t"):
    col_names = ["user", "item", "ts"]
    df = pd.read_csv(filname, sep=sep, header=None, names=col_names, engine="python")
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    df = df.sort_values(by=["ts"])
    df["year"], df["month"], df["day"], df["dayofweek"], df["dayofyear"], df["week"] = (
        zip(
            *df["ts"].map(
                lambda x: [
                    x.year,
                    x.month,
                    x.day,
                    x.dayofweek,
                    x.dayofyear,
                    x.isocalendar()[1],
                ]
            )
        )
    )
    df["year"] -= df["year"].min()
    df["year"] /= df["year"].max() if df["year"].max() > 0 else 1
    df["month"] /= 12
    df["day"] /= 31
    df["dayofweek"] /= 7
    df["dayofyear"] /= 365
    df["week"] /= 4
    DATEINFO = {}
    UsersDict = {}
    for index, row in df.iterrows():
        userid = int(row["user"])
        itemid = int(row["item"])
        year = row["year"]
        month = row["month"]
        day = row["day"]
        dayofweek = row["dayofweek"]
        dayofyear = row["dayofyear"]
        week = row["week"]
        DATEINFO[(userid, itemid)] = [year, month, day, dayofweek, dayofyear, week]
    return df, DATEINFO


def get_ItemDataGames(itemnum):
    ItemFeatures = load_data("./Data/Video_Games_feat.dat")
    ItemFeatures = np.vstack(
        (np.zeros((1, ItemFeatures.shape[1]), dtype=ItemFeatures.dtype), ItemFeatures)
    )
    return ItemFeatures
