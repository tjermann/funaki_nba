import os
import pandas as pd
import numpy as np
import re
from dask import bag as db
from .sportsref import return_schedule, return_pbp, clean_multigame_features
from .misc import flatten
import datetime
import time

def process_boxscore_id(boxscore_id):
    """Takes a boxscore ID and returns the play-by-play data (without lineup
    data), excluding overtimes.

    :param boxscore_id: A string containing a game's boxscore ID.
    :returns: a DataFrame of regulation play-by-play data.
    """
    try:
        year = int(boxscore_id[:4])
        month = int(boxscore_id[4:6])
        if month > 9:
            season = str(int(year) + 1)
        else:
            season = str(year)
        df = return_pbp(boxscore_id, season, dense_lineups=True, sparse_lineups=False)
        print("Successful: %s" % (boxscore_id))
        return df
    except Exception as e:
        print("Failed: %s" % (boxscore_id))
        return None


def get_new_pbp_data(pbp_br, year, schedule_col=None):

    today_ts = datetime.date.today()
    today_br = str(today_ts.year) + str(today_ts.month).zfill(2) + str(today_ts.day).zfill(2) + '0AAA'

    process_boxscore_id('201606190GSW')

    if schedule_col is not None:
        prev_ts = datetime.date.today() - datetime.timedelta(days=14)
        prev_br = str(prev_ts.year) + str(prev_ts.month).zfill(2) + str(prev_ts.day).zfill(2) + '0AAA'

        cursor = schedule_col.find({"$and" : [ { "id" : { "$gt" : prev_br } }, { "id" : { "$lt" : today_br } } ] }, {'id':1.0})
        schedule_df = pd.DataFrame([flatten(doc) for doc in cursor])
        boxscore_ids = schedule_df['id'].values

    else:
        months = "10,11,12,1,2,3,4,5,6,7,8"
        print(months)

        schedule = return_schedule(year, months)
        schedule['br_id'] = schedule['date_game'].map(lambda _date: _date.replace('=','&').split('&')[5]+_date.replace('=','&').split('&')[1].zfill(2)+_date.replace('=','&').split('&')[3].zfill(2))
        schedule['br_id'] = schedule['br_id']+'0'+schedule['home_team_name']
        boxscore_ids = schedule.br_id.values
        boxscore_ids = [game for game in boxscore_ids if game == game]
    if pbp_br.count_documents({"season": int(year)}) > 0:
        cursor = pbp_br.find({"season": int(year)}, {"boxscore_id": 1.0})
        pbp_br_games = pd.DataFrame([doc for doc in cursor])["boxscore_id"]
    else:
        pbp_br_games = pd.DataFrame()

    new_games = [
        game
        for game in boxscore_ids
        if game not in pbp_br_games.values and game == game and game != None and game < today_br
    ]
    print(new_games)
    
    if len(new_games) > 0:

        bsids_bag = db.from_sequence(new_games, npartitions=4)
        dfs_bag = bsids_bag.map(process_boxscore_id)
        dfs = dfs_bag.compute()
        filt_dfs = [
            df.drop(["nan_in"], axis=1, errors="ignore") for df in dfs if df is not None
        ]
        if len(filt_dfs)>0:
            df = pd.concat(filt_dfs)
            clean_df = clean_multigame_features(df)
            return clean_df
        else:
            print("No new games to import")
            return pd.DataFrame()

    else:
        print("No new games to import")
        return pd.DataFrame()

def get_new_schedule(year, all_months=False):

    if all_months:
        months=None

    else:
        today_ts = datetime.date.today()

        today_month = today_ts.month

        months =f"{str(today_month)}"

    print(months)

    schedule = return_schedule(year, months)

    schedule['br_id'] = schedule['date_game'].map(lambda _date: _date.replace('=','&').split('&')[5]+_date.replace('=','&').split('&')[1].zfill(2)+_date.replace('=','&').split('&')[3].zfill(2))
    schedule['br_id'] = schedule['br_id']+'0'+schedule['home_team_name']

    schedule['date'] =  schedule['br_id'].map(lambda br_id: br_id[:4]) + '-' + schedule['br_id'].map(lambda br_id: br_id[4:6])  + '-' + schedule['br_id'].map(lambda br_id: br_id[6:8])

    return schedule

def union(a, b):
    """ return the union of two lists """
    return list(set(a) | set(b))


def difference(a, b):
    """ return the difference of two lists """
    return list(set(a) - set(b))


def create_game_dict(boxscore_id, pbp_data):
    sparse_cols = pbp_data.columns[pbp_data.columns.str.contains("_in")]
    dense_cols = pbp_data.columns[pbp_data.columns.str.contains("player")]
    hm_dense_cols = pbp_data.columns[pbp_data.columns.str.contains("hm_player")]
    hm_dict = {"hm_player{}".format(i): "player{}".format(i) for i in range(1, 6)}
    aw_dense_cols = pbp_data.columns[pbp_data.columns.str.contains("aw_player")]
    aw_dict = {"aw_player{}".format(i): "player{}".format(i) for i in range(1, 6)}

    play_cols = pbp_data.columns[pbp_data.columns.str.contains("is_")]

    game_cols = [
        "season",
        "year",
        "month",
        "day",
        "boxscore_id",
        "home_abbr",
        "away_abbr",
    ]

    data_cols = difference(
        pbp_data.columns,
        union(union(union(play_cols, sparse_cols), dense_cols), game_cols),
    )

    tmp_df = pbp_data[pbp_data.boxscore_id == boxscore_id]

    game_data = tmp_df[game_cols].drop_duplicates().dropna(axis=1).to_dict("records")[0]

    game_data["date"] = (
        game_data["boxscore_id"][:4]
        + "-"
        + game_data["boxscore_id"][4:6]
        + "-"
        + game_data["boxscore_id"][6:8]
    )

    game_data["plays"] = []

    for play_num in range(len(tmp_df)):

        play = tmp_df.iloc[play_num]

        play_data = (
            pd.DataFrame(play[data_cols].dropna())
            .T.apply(pd.to_numeric, errors="ignore")
            .to_dict("records")[0]
        )

        play_data["home_players"] = pd.DataFrame(
            play[hm_dense_cols].rename(hm_dict)
        ).T.to_dict("records")[0]
        play_data["away_players"] = pd.DataFrame(
            play[aw_dense_cols].rename(aw_dict)
        ).T.to_dict("records")[0]

        play_data["play_type"] = (
            pd.DataFrame(play[play_cols])
            .T.apply(pd.to_numeric, errors="ignore")
            .to_dict("records")[0]
        )

        game_data["plays"].append(play_data)

    return game_data


def create_multigame_dict(pbp_data, n_jobs=1):

    pbp_data.rename({"home": "home_abbr", "away": "away_abbr"}, axis=1, inplace=True)

    multigame_dict = []

    bsids_bag = db.from_sequence(pbp_data.boxscore_id.unique(), npartitions=n_jobs)
    dict_bag = bsids_bag.map(create_game_dict, pbp_data=pbp_data)
    multigame_dict = dict_bag.compute()

    return multigame_dict
