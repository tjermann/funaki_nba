import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from pymongo import MongoClient
from src.pbp_update import pull_and_process_pbp
from src.forward_filtering import (
    forward_filter_season,
    return_apm_prior,
)
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import MONGODB_CONFIG, NBA_SEASON

def main(uri, start_date, end_date, prior_start_date, new_season=False):
    client = MongoClient(uri)
    database = client[MONGODB_CONFIG['database']]
    funaki_ratings = database[MONGODB_CONFIG['collections']['funaki_ratings']]
    funaki_ratings = database["funaki_ratings"]
    funaki_ratings.create_index("id")
    play_by_play = database[MONGODB_CONFIG['collections']['play_by_play']]

    today_ts = datetime.date.today()

    if new_season:
        look_back = 350
    else:
        look_back = 90
        
    if not prior_start_date:
    	prior_start_date = (today_ts - datetime.timedelta(days=look_back)).strftime("%Y-%m-%d")
    if not start_date:
    	start_date = (today_ts - datetime.timedelta(days=14)).strftime("%Y-%m-%d")
    if not end_date:
    	end_date = (today_ts + datetime.timedelta(days=5)).strftime("%Y-%m-%d")
    today = today_ts.strftime("%Y-%m-%d")

    print("Collecting APM Prior for dates %s to %s" % (prior_start_date, today))
    print(look_back)
    # apm_mean_df = pd.DataFrame()
    # apm_scale_df = pd.DataFrame()
    apm_mean_df, apm_scale_df = return_apm_prior(funaki_ratings, prior_start_date, today)

    print("Prior_DF")
    print(apm_mean_df.shape)
    print(apm_mean_df.index[-5:])

    if new_season:
        print("Adding new season prior")
        new_year = str(int(apm_mean_df.index[-1][:4]) + 1)

        apm_prior = apm_mean_df.iloc[-1] * 0.9
        apm_prior.name = new_year + "_prior"

        apm_mean_df = apm_mean_df.append(apm_prior)

        apm_scale_prior = (apm_scale_df.iloc[-1] * 1.25).clip(0.0, 0.02)
        apm_scale_prior.name = new_year + "_prior"

        apm_scale_df = apm_scale_df.append(apm_scale_prior)

    print("Collecting PBP data for dates %s to %s" % (start_date, end_date))

    X_apm, X_hca, y, ids = pull_and_process_pbp(play_by_play, start_date, end_date, reg_only=False)

    new_ids = [
        game_id for game_id in ids.unique() if game_id not in list(apm_mean_df.index)
    ]
    new_indx = np.where(ids.isin(new_ids))

    X_apm = X_apm.iloc[new_indx]
    X_hca = X_hca.iloc[new_indx]
    y = y[new_indx]
    ids = ids.iloc[new_indx]

    print("New Games")

    print(X_apm.shape)
    print(y.shape)
    print(ids.unique())

    if X_apm.shape[0] > 0:
        prior_len = apm_mean_df.shape[0]

        X_apm = X_apm.astype(np.float32)
        X_hca = X_hca.astype(np.float32)

        y = y.reshape(-1, 1).astype("float32")

        X_apm_tensor = tf.convert_to_tensor(X_apm)
        X_hca_tensor = tf.convert_to_tensor(X_hca)

        print(apm_mean_df.shape)
        print(apm_scale_df.shape)

        apm_mean_df = apm_mean_df.loc[:, apm_mean_df.columns == apm_mean_df.columns]
        apm_scale_df = apm_scale_df.loc[:, apm_scale_df.columns == apm_scale_df.columns]

        print(apm_mean_df.shape)
        print(apm_scale_df.shape)

        apm_mean_df, apm_scale_df, hca_mean_df, q_samples_apm_ = forward_filter_season(
            X_apm,
            X_hca,
            y,
            ids,
            init_scale=0.02,
            scale_update=0.0,
            apm_mean_df=apm_mean_df,
            apm_scale_df=apm_scale_df,
        )
        date = apm_scale_df.index[-1][:9]
        apm_mean_df = apm_mean_df.append(pd.DataFrame(index=[date + "END"]))
        apm_scale_df = apm_scale_df.append(pd.DataFrame(index=[date + "END"]))

        new_apm_mean_df = apm_mean_df.iloc[prior_len - 1 :].shift(1).dropna()
        new_apm_scale_df = apm_scale_df.iloc[prior_len - 1 :].shift(1).dropna()
        # new_apm_mean_df = apm_mean_df.shift(1).dropna()
        # new_apm_scale_df = apm_scale_df.shift(1).dropna()

        game_list = []

        for game in new_apm_mean_df.index:

            if "prior" in game:
                continue

            date = game[:4] + "-" + game[4:6] + "-" + game[6:8]

            apm_vals = new_apm_mean_df.loc[game]
            apm_scales = new_apm_scale_df.loc[game]

            player_list = []
            team_list = []

            for player in apm_vals.index:

                player_apm = apm_vals.loc[player]
                player_split = player.split("_")

                br_id = player_split[0]
                apm_type = player_split[1]

                if br_id == "RP":
                    continue

                player_scale = apm_scales.loc[player]

                player_data = {
                    "player_id": br_id,
                    "type": apm_type,
                    "value": float(player_apm),
                    "scale": float(player_scale),
                }

                player_list.append(player_data)

            game_info = {
                "id": game,
                "date": date,
                "players": player_list,
            }
            funaki_ratings.find_one_and_update(
                {"id": game_info["id"]}, {"$set": game_info}, upsert=True
            )
            print("Inserted {}".format(game))

        print("{} games Inserted".format(new_apm_mean_df.shape[0]))

    else:
        print("No new games to forward filter through")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments required for APM forward filtering."
    )

    parser.add_argument("--uri", default="mongodb://localhost:27017/")
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--prior_start_date", default=None)
    parser.add_argument("--new_season", action="store_true", help="Set if first run of new season.")
    args = parser.parse_args()
    main(args.uri, args.start_date, args.end_date, args.prior_start_date, args.new_season)


    # dates=[
    #     '2021-07-15',
    #     '2022-07-15',
    # ]

    # for i in range(0, len(dates)-2):
    #     prior_start_date = dates[i]
    #     start_date = dates[i+1]
    #     end_date = dates[i+2]

    # main('2022-07-15', '2023-07-15', '2022-02-15', True)