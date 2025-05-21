from pymongo import MongoClient
from urllib.parse import quote_plus
from src.pbp_update import get_new_pbp_data, create_multigame_dict
import argparse
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import MONGODB_CONFIG, NBA_SEASON

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments required to update basketball reference pbp collection"
    )
    parser.add_argument(
        "--year",
        default=str(NBA_SEASON),
        help="Basketball-Reference season year.  Year in which season concludes",
    )
    parser.add_argument("--uri", default=MONGODB_CONFIG['uri'], help="MongoDB URI")
    parser.add_argument("--backfill", action="store_true", help="Backfill pbp data")

    args = parser.parse_args()

    client = MongoClient(args.uri)
    database = client[MONGODB_CONFIG['database']]
    pbp = database[MONGODB_CONFIG['collections']['pbp']]
    pbp.create_index("season")

    if args.backfill:
        schedule = None
        years = "2016,2017,2018,2019,2020,2021,2022,2023,2024,2025"

        print("\n\nBackfilling pbp data for years: {}\n".format(years))
        print("This may take a while...It may be required to update the hard coded years list in insert_pbp_data.py and run in chunks")
        print("Additionally you may hit throttling limits from basketball reference")
        print("If you do, that rate limiting typically resets in 12-24 hours\n")

    else:
        years = args.year
        schedule = database[MONGODB_CONFIG['collections']['schedule']]

    years = years.split(',')

    for year in years:
        print(year)
        pbp_data = get_new_pbp_data(pbp, year, schedule)

        if len(pbp_data) > 0:
            # n_jobs may need to be limited due to memory constraints when
            # parallelizing over larger time horizons
            multigame_dict = create_multigame_dict(pbp_data, n_jobs=1)

            # pbp.insert_many(multigame_dict)
            print("{} games inserted successfully".format(len(multigame_dict)))
        else:
            print("No new games to insert")
