import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import sts
from .dynamic_apm import (
    SparseDynamicAPMStateSpaceModel,
    SparseDynamicAPM,
    build_apm_model,
)
from .process_pbp import pull_and_process_pbp

from .misc import flatten

def forward_filter_season(
    X_apm,
    X_hca,
    y,
    ids,
    q_samples_apm_=None,
    init_scale=0.01,
    scale_update=0.0,
    apm_mean_df=None,
    apm_scale_df=None,
):

    if apm_mean_df is None:
        apm_mean_df = pd.DataFrame()
    if apm_scale_df is None:
        apm_scale_df = pd.DataFrame()
    hca_mean_df = pd.DataFrame()

    if q_samples_apm_ is None:
        q_samples_apm_ = {}

    print('recentering')
    # Centering each possession to the league average of 1.0 pts per play
    y = y - 1.0

    start_id = ids.values[0]
    date = start_id[:4] + "-" + start_id[4:6] + "-" + start_id[6:8]

    ids.reset_index(drop=True, inplace=True)

    tf.compat.v1.reset_default_graph()

    game_ids = ids.unique()
    i = 1
    for game_id in game_ids:
        print("Game ID {}: {} out of {}".format(game_id, i, len(game_ids)))

        games = ids[ids == game_id].index

        # Filter apm/hca/y data to a single game
        game_apm = X_apm.iloc[games, :]
        game_apm = game_apm[game_apm.columns[game_apm.sum(axis=0) != 0]]

        game_hca = X_hca.iloc[games, :]

        game_apm_tensor = tf.convert_to_tensor(game_apm, dtype=tf.float32)
        game_hca_tensor = tf.convert_to_tensor(game_hca, dtype=tf.float32)

        game_y = y[games]

        # Initialize apm prior
        if apm_mean_df.shape[1] > 0:
            game_apm_means_prior = (
                # apm_mean_df.ffill().iloc[-1][game_apm.columns].fillna(0)
                apm_mean_df.ffill().iloc[-1].reindex(game_apm.columns).fillna(0)
            )

            game_apm_scales_prior = (
                apm_scale_df.ffill().iloc[-1].reindex(game_apm.columns).fillna(init_scale)
            )

            # Game update of scale_update
            game_scale_update = pd.Series(
                scale_update * np.ones(game_apm.shape[1]), index=game_apm.columns
            )
            game_apm_scales_prior += game_scale_update

            # Lag update for players who have not played - goal of setting scale = init_scale if it's been a year
            # since player last played
            time_since_last_game_scale = (
                apm_scale_df.bfill().isnull().sum().reindex(game_apm.columns).fillna(0)
                * init_scale
                / 10000.0
            )
            game_apm_scales_prior += time_since_last_game_scale

            # Set max scale to initial scale of init_scale
            game_apm_scales_prior = game_apm_scales_prior.map(
                lambda scale: min(scale, init_scale)
            )

        else:
            game_apm_means_prior = pd.Series(
                np.zeros(game_apm.shape[1]), index=game_apm.columns
            )
            game_apm_scales_prior = pd.Series(
                init_scale * np.ones(game_apm.shape[1]), index=game_apm.columns
            )

        game_apm_weights = tfd.MultivariateNormalDiag(
            loc=game_apm_means_prior.astype("float32"),
            scale_diag=game_apm_scales_prior.astype("float32"),
        )

        # Build ssm and forward filter
        tf.compat.v1.reset_default_graph()

        apm_model_forward = build_apm_model(game_y, game_apm_tensor, game_hca_tensor)

        hca_points = pd.Series(np.ones(X_hca.shape[1]) * 0.015)

        hca_std = hca_points * 0.0 + 0.001

        hca_means_post = hca_points.astype("float32")
        hca_scales_post = hca_std.astype("float32")

        hca_weights_mean = (
            np.ones((200, X_hca.shape[1])) * hca_means_post.values
        ).astype("float32")
        hca_weights_scale = np.vstack(
            [
                np.random.normal(loc=0.0, scale=hca_scales_post[i], size=(200,))
                for i in range(X_hca.shape[1])
            ]
        ).T.astype("float32")

        q_samples_apm_["hca/_weights"] = tf.convert_to_tensor(
            (hca_weights_mean + hca_weights_scale)
        )
        q_samples_apm_["observation_noise_scale"] = tf.convert_to_tensor(
            1.15 * np.ones((200,)) + np.random.normal(loc=0.0, scale=0.1, size=(200,)),
            dtype="float32",
        )
        q_samples_apm_["apm/_drift_scale"] = tf.convert_to_tensor(
            0.0001 * np.ones((200,))
            + np.random.normal(loc=0.0, scale=0.00001, size=(200,)),
            dtype="float32",
        )

        if i % 100 == 1:
            print("Inferred parameters:")
            for param in apm_model_forward.parameters:
                print(
                    "{}: {} +- {}".format(
                        param.name,
                        np.mean(q_samples_apm_[param.name], axis=0),
                        np.std(q_samples_apm_[param.name], axis=0),
                    )
                )

        apm_ssm = apm_model_forward.make_state_space_model(
            len(game_y), q_samples_apm_, initial_state_prior=game_apm_weights
        )

        filtered_vals = apm_ssm.forward_filter(game_y)

        (
            log_liks,
            filtered_apms,
            filtered_scales,
            pred_apms,
            pred_scales,
            observation_means,
            observation_scales,
        ) = filtered_vals

        filt_apms_array = tf.reduce_mean(filtered_apms, axis=0).numpy()
        apm_means_post = filt_apms_array[-1, :]

        filt_apms_scales_array = tf.reduce_mean(filtered_scales, axis=0).numpy()
        apm_scales_post = np.sqrt(np.diagonal(filt_apms_scales_array[-1, :, :]))

        # Append post apm mean and scale
        apm_mean_df = apm_mean_df.append(
            pd.DataFrame(apm_means_post, index=game_apm.columns, columns=[game_id]).T
        )
        apm_scale_df = apm_scale_df.append(
            pd.DataFrame(apm_scales_post, index=game_apm.columns, columns=[game_id]).T
        )
        hca_mean_df = hca_mean_df.append(
            pd.Series(
                np.mean(q_samples_apm_["hca/_weights"], axis=0),
                index=X_hca.columns,
                name=game_id,
            )
        )

        i += 1

    apm_mean_df = apm_mean_df.ffill().fillna(0.0)
    apm_scale_df = apm_scale_df.ffill().fillna(init_scale)
    hca_mean_df = hca_mean_df.ffill().fillna(0.015)

    return apm_mean_df, apm_scale_df, hca_mean_df, q_samples_apm_

def return_apm_prior(collection, start_date, end_date):
    """

    Pull APM values for year and process into multi-keyed dictionary {(GameID, playerID): value}

    params:
    collection: mongo collection
    year: string value for year ending


    """

    pipeline = [
        {u"$match": {"date": {"$gte": start_date, "$lte": end_date}}},
        {u"$unwind": {u"path": u"$players"}},
        {
            u"$addFields": {
                u"player_id": {
                    u"$concat": [u"$players.player_id", u"_", u"$players.type"]
                },
                u"apm": u"$players.value",
                u"scale": u"$players.scale",
            }
        },
        {u"$project": {u"id": 1.0, u"player_id": 1.0, u"apm": 1.0, u"scale": 1.0}},
    ]

    cursor = collection.aggregate(pipeline, allowDiskUse=False)

    apm_prior = pd.DataFrame([flatten(doc) for doc in cursor])

    game_ids = list(apm_prior["id"].unique())

    players = (
        apm_prior[["id", "player_id", "apm"]]
        .groupby("id")["player_id"]
        .apply(lambda df: df.reset_index(drop=True))
        .unstack()
        .iloc[-1]
        .values
    )

    apm_mean_df = (
        apm_prior[["id", "player_id", "apm"]]
        .groupby("id")["apm"]
        .apply(lambda df: df.reset_index(drop=True))
        .unstack()
    )
    apm_mean_df.columns = players
    apm_mean_df = apm_mean_df.loc[game_ids, :]

    apm_scale_df = (
        apm_prior[["id", "player_id", "scale"]]
        .groupby("id")["scale"]
        .apply(lambda df: df.reset_index(drop=True))
        .unstack()
    )
    apm_scale_df.columns = players
    apm_scale_df = apm_scale_df.loc[game_ids, :]

    return apm_mean_df, apm_scale_df
