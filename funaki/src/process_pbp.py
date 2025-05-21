import pandas as pd
import numpy as np
from bson.son import SON
import datetime
import re

from joblib import parallel_backend, Parallel, delayed

from .misc import flatten

PLAYER_RE = r"\w{0,7}\d{2}"

def get_pbp_data(pbp_br, start_date, end_date, reg_only=False):

    """

    Functon for returning play by play data from mongo and processing

    pbp_br: mongo connection string to play by play collection
    start_date: start date for play by play data (inclusive) in form YYYY-MM-DD
    end_date: end date for play by play data (inclusive) in form YYYY-MM-DD


    returns: dataframe with play by play data


    Notes:
    ### Returns sparse columns as 1 for home, 0 for away to stay consistent with existing functionality
    ### Needs to be further processed to 1 for off, 0 for def


    """

    pipeline = [
        {u"$match": {u"date": {u"$gte": start_date, u"$lte": end_date}}},
        {u"$sort": SON([ (u"date", 1) ])},
        {u"$unwind": "$plays"},
    ]

    if reg_only:
        pipeline.insert(0, {u"$match": {u"season_type": u"reg"}})

    cursor = pbp_br.aggregate(pipeline, allowDiskUse=False)

    df = pd.DataFrame([flatten(doc) for doc in cursor])

    if df.shape[0] == 0:
        print("No Games in given date")
        return pd.DataFrame()

    df.rename(
        columns={
            col: col.replace("plays_", "") for col in df.columns if "plays_" in col
        },
        inplace=True,
    )
    df.rename(
        columns={
            col: col.replace("play_type_", "")
            for col in df.columns
            if "play_type_" in col
        },
        inplace=True,
    )

    df.rename(
        columns={
            col: col.replace("away_players_", "aw_")
            for col in df.columns
            if "away_players_" in col
        },
        inplace=True,
    )
    df.rename(
        columns={
            col: col.replace("home_players_", "hm_")
            for col in df.columns
            if "home_players_" in col
        },
        inplace=True,
    )
    home_players = [col for col in df.columns if "hm_player" in col]
    away_players = [col for col in df.columns if "aw_player" in col]

    players = home_players + away_players

    unique_players = []

    for player in players:

        unique_players = list(set((unique_players + list(df[player].values))))

    unique_players = [plyr for plyr in unique_players if plyr == plyr]

    unique_players = [plyr + "_in" for plyr in sorted(unique_players)]
    sparse_players = np.zeros((df.shape[0], len(unique_players)))

    for team in ["hm", "aw"]:
        for i in range(1, 6):

            pos = pd.get_dummies(df[[team + "_player" + str(i)]])
            pos.rename(
                columns={plyr: plyr.split("_")[2] + "_in" for plyr in pos.columns},
                inplace=True,
            )
            pos = pos.reindex(sorted(pos.columns), axis=1)

            ### If sparse columns to be returned as HOME/AWAY (1/-1)
            pos = pd.DataFrame(
                pos.values * (int(team == "hm") - 0.5) * 2.0, columns=pos.columns
            )

            ### If sparse columns to be returned as OFF/DEF (1/-1)
            # pos = pd.DataFrame(pos.values * ((df.hm_off.astype(int)-0.5)*2.0 * (int(team=='hm')-0.5)*2.0).values.reshape(-1,1), columns = pos.columns)

            pos_indices = [
                indx
                for indx, plyr in enumerate(sorted(unique_players))
                if plyr in pos.columns
            ]

            sparse_players[:, pos_indices] += pos.values

    sparse_players_df = pd.DataFrame(
        sparse_players, columns=unique_players, index=df.index
    )

    df = df.merge(sparse_players_df, left_index=True, right_index=True)

    return df

def combine_stat(stat_func, pbp, players, reps):

    pbp = stat_func(pbp, players, reps).fillna(0)

    return pbp


def get_off_plays(df, players, reps):
    def num_on_off(df, p):
        in_on_off = df["{}_in".format(p)] == (2 * df.hm_off - 1)
        return df.loc[in_on_off, "play_id"].nunique()

    on_floor = df.loc[(df.is_fga == 1.0) | (df.is_to == 1.0) | (df.is_pf == 1.0)]
    player_ops = pd.Series(
        {
            p: num_on_off(on_floor, p) if "{}_in".format(p) in df else 0.0
            for p in players
        }
    )
    rep_ops = np.sum(
        num_on_off(on_floor, p) if "{}_in".format(p) in df else 0.0 for p in reps
    )
    player_ops["RP"] = rep_ops
    return player_ops


def get_def_plays(df, players, reps):
    def num_on_def(df, p):
        in_on_def = df["{}_in".format(p)] == (-2 * df.hm_off + 1)
        return df.loc[in_on_def, "play_id"].nunique()

    on_floor = df.loc[(df.is_fga == 1.0) | (df.is_to == 1.0) | (df.is_pf == 1.0)]
    player_dps = pd.Series(
        {
            p: num_on_def(on_floor, p) if "{}_in".format(p) in df else 0.0
            for p in players
        }
    )
    rep_dps = np.sum(
        num_on_def(on_floor, p) if "{}_in".format(p) in df else 0.0 for p in reps
    )
    player_dps["RP"] = rep_dps
    return player_dps


def pull_and_process_pbp(pbp_br, start_date, end_date, return_game_data=False, reg_only=False):

    current_year = get_pbp_data(pbp_br, start_date, end_date, reg_only).loc[:, :]

    if current_year.shape[0] == 0:
        print("No games in date range")

        if return_game_data:
            return pd.DataFrame(), pd.DataFrame(), np.ones(0), pd.Series(), pd.DataFrame()

        else:
            return pd.DataFrame(), pd.DataFrame(), np.ones(0), pd.Series()

    all_pcols = sparse_lineup_cols(current_year)
    all_players = [col[:-3] for col in all_pcols]

    current_year["date"] = current_year.boxscore_id.map(lambda bid: bid[:-3])

    all_off_plays = combine_stat(get_off_plays, current_year, all_players, [])
    all_def_plays = combine_stat(get_def_plays, current_year, all_players, [])
    all_plays = all_off_plays + all_def_plays
    all_plays.sort_values(ascending=False, inplace=True)

    players = all_plays.index.values
    off_plays = all_off_plays[players]
    def_plays = all_def_plays[players]

    combined_df = clean_multigame_features(current_year)

    def on_off(df, p):
        col = "{}_in".format(p)
        if col in df.columns:
            return (df[col].astype(int) == (2 * df.hm_off.astype(int) - 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    def on_def(df, p):
        col = "{}_in".format(p)
        if col in df.columns:
            return (df[col].astype(int) == (-2 * df.hm_off.astype(int) + 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    print("Starting RAPM calculations")

    poss_end = combined_df.groupby("poss_id").tail(1)
    poss_end = poss_end.query("is_fga | is_to | is_pf_fta")
    poss_end.drop(["pts"], axis=1, inplace=True)

    points = combined_df.groupby("poss_id").pts.sum()
    poss_end = pd.merge(
        poss_end, points, left_on="poss_id", right_index=True, how="left"
    )

    poss_end.reset_index(drop=True, inplace=True)

    print("computing off_df")
    off_df = pd.DataFrame({"{}_orapm".format(p): on_off(poss_end, p) for p in players})

    off_df = pd.DataFrame(
        off_df.values / off_df.sum(axis=1).values.reshape(-1, 1) * 5.0,
        columns=off_df.columns,
    )

    print("computing def_df")
    def_df = pd.DataFrame({"{}_drapm".format(p): on_def(poss_end, p) for p in players})

    def_df = pd.DataFrame(
        def_df.values / def_df.sum(axis=1).values.reshape(-1, 1) * 5.0,
        columns=def_df.columns,
    )

    print("computing X")

    X_apm = pd.concat((off_df, -def_df), axis=1).fillna(0).reset_index(drop=True)

    print(X_apm.shape)
    X_apm.head()

    dates = poss_end.date
    ids = poss_end.boxscore_id

    home_one_hot = pd.get_dummies(poss_end.home_abbr).reset_index(drop=True).astype(int)

    X_hca = home_one_hot 

    hca_off = poss_end.hm_off
    hca_def = poss_end.hm_off - 1

    X_hca_D = pd.DataFrame(
        X_hca.values * hca_def.values.reshape(-1, 1),
        columns=[col + "_D" for col in X_hca.columns],
    )
    X_hca_O = pd.DataFrame(
        X_hca.values * hca_off.values.reshape(-1, 1),
        columns=[col + "_O" for col in X_hca.columns],
    )

    X_hca = pd.merge(X_hca_O, X_hca_D, right_index=True, left_index=True)

    X_hca.sort_index(axis=1, inplace=True)

    y = poss_end.pts.values

    season_min = poss_end.season.min()
    weights = np.where(poss_end.season == season_min, 1, 6)

    if return_game_data==True:

        return X_apm, X_hca, y, ids, poss_end[['hm_score','aw_score','secs_elapsed','quarter','hm_off']]
    else:
        return X_apm, X_hca, y, ids
    
def pull_and_process_stat(pbp_br, start_date, end_date, stat, return_game_data=False, reg_only=False):

    current_year = get_pbp_data(pbp_br, start_date, end_date, reg_only).loc[:, :]

    if current_year.shape[0] == 0:
        print("No games in date range")

        if return_game_data:
            return pd.DataFrame(), pd.DataFrame(), np.ones(0), pd.Series(), pd.DataFrame()

        else:
            return pd.DataFrame(), pd.DataFrame(), np.ones(0), pd.Series()

    all_pcols = sparse_lineup_cols(current_year)
    all_players = [col[:-3] for col in all_pcols]

    current_year["date"] = current_year.boxscore_id.map(lambda bid: bid[:-3])

    all_off_plays = combine_stat(get_off_plays, current_year, all_players, [])
    all_def_plays = combine_stat(get_def_plays, current_year, all_players, [])
    all_plays = all_off_plays + all_def_plays
    all_plays.sort_values(ascending=False, inplace=True)

    players = all_plays.index.values
    off_plays = all_off_plays[players]
    def_plays = all_def_plays[players]

    combined_df = clean_multigame_features(current_year)

    def on_off(df, p):
        col = "{}_in".format(p)
        if col in df.columns:
            return (df[col].astype(int) == (2 * df.hm_off.astype(int) - 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    def on_def(df, p):
        col = "{}_in".format(p)
        if col in df.columns:
            return (df[col].astype(int) == (-2 * df.hm_off.astype(int) + 1)).astype(int)
        else:
            return np.zeros(df.shape[0])

    print("Starting RAPM calculations")

    for feat in ['is_oreb','is_dreb','is_assist']:
        combined_df[feat] = combined_df[feat].astype(int)

    combined_df[stat] = combined_df[f"is_{stat}"]

    poss_end = combined_df.groupby("poss_id").tail(1)
    poss_end = poss_end.query("is_fga | is_to | is_pf_fta")
    poss_end.drop(["pts"], axis=1, inplace=True)
    poss_end.drop([stat], axis=1, inplace=True)

    stat_sum = combined_df.groupby("poss_id")[stat].sum()
    poss_end = pd.merge(
        poss_end, stat_sum, left_on="poss_id", right_index=True, how="left"
    )

    points = combined_df.groupby("poss_id").pts.sum()
    poss_end = pd.merge(
        poss_end, points, left_on="poss_id", right_index=True, how="left"
    )

    poss_end.reset_index(drop=True, inplace=True)

    print("computing off_df")
    off_df = pd.DataFrame({"{}_orapm".format(p): on_off(poss_end, p) for p in players})

    off_df = pd.DataFrame(
        off_df.values / off_df.sum(axis=1).values.reshape(-1, 1) * 5.0,
        columns=off_df.columns,
    )

    print("computing def_df")
    def_df = pd.DataFrame({"{}_drapm".format(p): on_def(poss_end, p) for p in players})

    def_df = pd.DataFrame(
        def_df.values / def_df.sum(axis=1).values.reshape(-1, 1) * 5.0,
        columns=def_df.columns,
    )

    print("computing X")
    X_apm = pd.concat((off_df, -def_df), axis=1).fillna(0).reset_index(drop=True)

    print(X_apm.shape)
    X_apm.head()

    dates = poss_end.date
    ids = poss_end.boxscore_id

    home_one_hot = pd.get_dummies(poss_end.home_abbr).reset_index(drop=True).astype(int)

    X_hca = home_one_hot 

    hca_off = poss_end.hm_off
    hca_def = poss_end.hm_off - 1

    X_hca_D = pd.DataFrame(
        X_hca.values * hca_def.values.reshape(-1, 1),
        columns=[col + "_D" for col in X_hca.columns],
    )
    X_hca_O = pd.DataFrame(
        X_hca.values * hca_off.values.reshape(-1, 1),
        columns=[col + "_O" for col in X_hca.columns],
    )

    X_hca = pd.merge(X_hca_O, X_hca_D, right_index=True, left_index=True)

    X_hca.sort_index(axis=1, inplace=True)

    y = poss_end[stat].values

    season_min = poss_end.season.min()
    weights = np.where(poss_end.season == season_min, 1, 6)

    if return_game_data==True:

        return X_apm, X_hca, y, ids, poss_end[['hm_score','aw_score','secs_elapsed','quarter','hm_off','pts',stat]]
    else:
        return X_apm, X_hca, y, ids

def sparse_lineup_cols(df):
    regex = "{}_in".format(PLAYER_RE)
    return [c for c in df.columns if re.match(regex, c)]


def get_off_plays(df, players, reps):
    def num_on_off(df, p):
        in_on_off = df["{}_in".format(p)] == (2 * df.hm_off - 1)
        return df.loc[in_on_off, "play_id"].nunique()

    on_floor = df.loc[(df.is_fga == 1.0) | (df.is_to == 1.0) | (df.is_pf == 1.0)]
    player_ops = pd.Series(
        {
            p: num_on_off(on_floor, p) if "{}_in".format(p) in df else 0.0
            for p in players
        }
    )
    rep_ops = np.sum(
        num_on_off(on_floor, p) if "{}_in".format(p) in df else 0.0 for p in reps
    )
    player_ops["RP"] = rep_ops
    return player_ops


def get_def_plays(df, players, reps):
    def num_on_def(df, p):
        in_on_def = df["{}_in".format(p)] == (-2 * df.hm_off + 1)
        return df.loc[in_on_def, "play_id"].nunique()

    on_floor = df.loc[(df.is_fga == 1.0) | (df.is_to == 1.0) | (df.is_pf == 1.0)]
    player_dps = pd.Series(
        {
            p: num_on_def(on_floor, p) if "{}_in".format(p) in df else 0.0
            for p in players
        }
    )
    rep_dps = np.sum(
        num_on_def(on_floor, p) if "{}_in".format(p) in df else 0.0 for p in reps
    )
    player_dps["RP"] = rep_dps
    return player_dps


def clean_features(df):
    """Fixes up columns of the passed DataFrame, such as casting T/F columns to
    boolean and filling in NaNs for team and opp.

    :param df: DataFrame of play-by-play data.
    :returns: Dataframe with cleaned columns.
    """
    df = pd.DataFrame(df)

    bool_vals = set([True, False, None, np.nan])
    sparse_cols = sparse_lineup_cols(df)
    for col in df:

        # make indicator columns boolean type (and fill in NaNs)
        if set(df[col].unique()[:5]) <= bool_vals:
            df[col] = df[col] == True

        # fill NaN's in sparse lineup columns to 0
        elif col in sparse_cols:
            df[col] = df[col].fillna(0)

    # fix free throw columns on technicals
    df.loc[df.is_tech_fta, ["fta_num", "tot_fta"]] = 1

    # fill in NaN's/fix off_team and def_team columns
    df.off_team.fillna(method="bfill", inplace=True)
    df.def_team.fillna(method="bfill", inplace=True)
    df.off_team.fillna(method="ffill", inplace=True)
    df.def_team.fillna(method="ffill", inplace=True)

    return df


def clean_multigame_features(df):

    df = pd.DataFrame(df)
    if df.index.value_counts().max() > 1:
        df.reset_index(drop=True, inplace=True)

    df = clean_features(df)

    # if it's many games in one DataFrame, make poss_id and play_id unique
    for col in ("play_id", "poss_id"):
        df[col] = df[col].fillna(method="bfill")
        df[col] = df[col].fillna(method="ffill")
        diffs = df[col].astype(int).diff().fillna(0)
        if (diffs < 0).any():
            new_col = np.cumsum(diffs.astype(bool))
            df.eval("{} = @new_col".format(col), inplace=True)

    return df


def combine_stat(stat_func, pbp, players, reps):

    pbp = stat_func(pbp, players, reps).fillna(0)

    return pbp

def on_off(df, p):
    col = "{}_in".format(p)
    if col in df.columns:
        return (df[col].astype(int) == (2 * df.hm_off.astype(int) - 1)).astype(int)
    else:
        return np.zeros(df.shape[0])


def on_def(df, p):
    col = "{}_in".format(p)
    if col in df.columns:
        return (df[col].astype(int) == (-2 * df.hm_off.astype(int) + 1)).astype(int)
    else:
        return np.zeros(df.shape[0])


def process_players_possessions(
    pbp, start_date, end_date
):
    """Function for returning up to date lineup distribution and adding it to dict based on 5 most recent games

    pbp_br: mongo collection string for play by play collection
    start_date: start date for collecting games in form 'YYYY-MM-DD'
    end_date: end date for collecting games (ideally yesterdays date in live setting) in form 'YYYY-MM-DD'
    today: todays date in form 'YYYYMMDD0'
    lookback_window: number of games for lookback window for lineup distribution

    returns: nested dictionary of {team: {date: {player1: poss,
                                                 player2: poss,
                                                 etc....}}}


    """

    team_dict = {}

    print(start_date)
    print(end_date)

    current_year = get_pbp_data(pbp, start_date, end_date)

    all_pcols = sparse_lineup_cols(current_year)
    all_players = [col[:-3] for col in all_pcols]

    current_year["date"] = current_year.boxscore_id.map(lambda bid: bid[:-3])

    all_off_plays = combine_stat(get_off_plays, current_year, all_players, [])
    all_def_plays = combine_stat(get_def_plays, current_year, all_players, [])
    all_plays = all_off_plays + all_def_plays
    all_plays.sort_values(ascending=False, inplace=True)

    players = all_plays.index.values
    off_plays = all_off_plays[players]
    def_plays = all_def_plays[players]

    combined_df = clean_multigame_features(current_year)

    poss_end = combined_df.groupby("poss_id").tail(1)
    poss_end = poss_end.query("is_fga | is_to | is_pf_fta")
    poss_end.drop(["pts"], axis=1, inplace=True)

    points = combined_df.groupby("poss_id").pts.sum()
    poss_end = pd.merge(
        poss_end, points, left_on="poss_id", right_index=True, how="left"
    )

    poss_end.reset_index(drop=True, inplace=True)

    off_df = pd.DataFrame({"{}".format(p): on_off(poss_end, p) for p in players})

    off_df['boxscore_id'] = poss_end['boxscore_id'].values
    off_df['home'] = np.where(poss_end["off_team"] == poss_end['home_abbr'], 1, 0)

    pace = off_df[['boxscore_id','home']]
    pace['quarter'] = poss_end['quarter'].values

    total_possessions = pace.copy()
    total_possessions.drop(['quarter'], axis=1, inplace=True)
    total_possessions = total_possessions.groupby('boxscore_id').count()['home']/2.0

    pace = pace.loc[pace['quarter'] <= 4]
    pace.drop(['quarter'], axis=1, inplace=True)
    pace = pace.groupby('boxscore_id').count()['home']/2.0

    off_df.drop(['RP'], axis=1)
    off_df = off_df.groupby(['boxscore_id','home']).sum()

    return off_df, pace, total_possessions

