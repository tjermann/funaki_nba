from pymongo import MongoClient
from urllib.parse import quote_plus
from src.pbp_update import get_new_schedule
from src.sportsref import parse_table
import argparse
import os
import sys
import os

from datetime import datetime, timedelta

from pyquery import PyQuery as pq
import requests

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from config import MONGODB_CONFIG, NBA_SEASON

month_map = {
            '1':'january',
            '2':'february',
            '3':'march',
            '4':'april',
            '5':'may',
            '6':'june',
            '7':'july',
            '8':'august',
            '9':'september',
            '10':'october',
            '11':'november',
            '12':'december',
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Arguments required to update basketball reference pbp collection"
    )
    parser.add_argument(
        "--year",
        default="2025",
        help="Basketball-Reference season year.  Year in which season concludes",
    )
    parser.add_argument(
        "--playoff_start",
        default='2025-04-15',
        help="Basketball-Reference season year.  Year in which season concludes",
    )
    parser.add_argument("--uri", default="mongodb://localhost:27017/")

    args = parser.parse_args()

    client = MongoClient(args.uri)
    database = client[MONGODB_CONFIG['database']]
    collection = database[MONGODB_CONFIG['collections']['schedule']]
    collection.create_index("id")
    collection.create_index("date")

    day = datetime.now().day
    month = datetime.now().month

    if len(args.year.split(',')) == 1:
        # schedule = get_new_schedule(args.year)

        if day < 3:
            if month == 1:
                month_nums = [12,1]
            else:
                month_nums = [month-1,1]
        elif day >27:
            if month == 12:
                month_nums = [12,1]
            else:
                month_nums = [month,month+1]
        else:
            month_nums = [month]

    
        for num in month_nums:
            month = month_map[str(num)]

            response = requests.get(f'https://www.basketball-reference.com/leagues/NBA_{args.year}_games-{month}.html')
            html = response.text
            html = html.replace('<!--', '').replace('-->', '')
            doc = pq(html)
            table = doc('table#schedule')
            schedule = parse_table(table)

            schedule['br_id'] = schedule['date_game'].map(lambda _date: _date.replace('=','&').split('&')[5]+_date.replace('=','&').split('&')[1].zfill(2)+_date.replace('=','&').split('&')[3].zfill(2))
            schedule['br_id'] = schedule['br_id']+'0'+schedule['home_team_name']

            schedule['date'] =  schedule['br_id'].map(lambda br_id: br_id[:4]) + '-' + schedule['br_id'].map(lambda br_id: br_id[4:6])  + '-' + schedule['br_id'].map(lambda br_id: br_id[6:8])
            schedule.loc[:, 'is_playoffs'] = False
            schedule.loc[schedule.date > args.playoff_start, 'is_playoffs'] = True

            if len(schedule) > 0:
                schedule['game_start_time'] = schedule['game_start_time'].fillna('7:00p')
                schedule['am_pm'] = schedule['game_start_time'].map(lambda _time: _time[-1])
                schedule.loc[schedule['am_pm'] == 'p','game_start_time'] = schedule.loc[schedule['am_pm'] == 'p','game_start_time'].map(lambda _time: str(int(_time[:-1].split(':')[0])+12) + ':' + _time[:-1].split(':')[1])
                schedule.loc[schedule['am_pm'] == 'a','game_start_time'] = schedule.loc[schedule['am_pm'] == 'a','game_start_time'].map(lambda _time: _time[:-1])

                for _, game in schedule.iterrows():
                    doc = {
                        'id':game['br_id'],
                        'date':game['date'],
                        'time':game['game_start_time'],
                        'home':game['home_team_name'],
                        'away':game['visitor_team_name'],
                        'home_points':game['home_pts'],
                        'away_points':game['visitor_pts'],
                        'overtimes':game['overtimes'],
                        'attendance':game['attendance'],
                        'arena':game['arena_name'],
                        'remarks':game['game_remarks'],
                        'is_playoffs':game['is_playoffs'],
                    }

                    collection.find_one_and_update({'id':game['br_id']},
                                                {'$set': doc}, upsert=True)
                
                print("{} games inserted successfully".format(len(schedule)))
            else:
                print("No new games to insert")

    else:
        years = args.year.split(',')

        for year in years:
            print(year)
            schedule = get_new_schedule(year, all_months=True)

            if len(schedule) > 0:
                for _, game in schedule.iterrows():
                    doc = {
                        'id':game['br_id'],
                        'date':game['date'],
                        'time':game['game_start_time'],
                        'home':game['home_team_name'],
                        'away':game['visitor_team_name'],
                        'home_points':game['home_pts'],
                        'away_points':game['visitor_pts'],
                        'overtimes':game['overtimes'],
                        'attendance':game['attendance'],
                        'arena':game['arena_name'],
                        'remarks':game['game_remarks'],
                        'is_playoffs':game['is_playoffs'],
                    }

                    collection.find_one_and_update({'id':game['br_id']},
                                                   {'$set': doc}, upsert=True)

                print("{} games inserted successfully".format(len(schedule)))
            else:
                print("No new games to insert")