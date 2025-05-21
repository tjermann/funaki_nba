import time
import requests
import ctypes
import threading
import multiprocessing
from past.builtins import basestring

import pandas as pd
import numpy as np
from pyquery import PyQuery as pq
import re

THROTTLE_DELAY = 0.5
BASE_URL = 'https://www.basketball-reference.com'

throttle_thread_lock = threading.Lock()
throttle_process_lock = multiprocessing.Lock()
last_request_time = multiprocessing.Value(ctypes.c_longdouble, time.time() - 10 * THROTTLE_DELAY)

PLAYER_RE = r'\w{0,7}(?:\w{1})\d{2}'

HM_LINEUP_COLS = ['hm_player{}'.format(i) for i in range(1, 6)]
AW_LINEUP_COLS = ['aw_player{}'.format(i) for i in range(1, 6)]
ALL_LINEUP_COLS = AW_LINEUP_COLS + HM_LINEUP_COLS

def rel_url_to_id(url):
    """Converts a relative URL to a unique ID.

    Here, 'ID' refers generally to the unique ID for a given 'type' that a
    given datum has. For example, 'BradTo00' is Tom Brady's player ID - this
    corresponds to his relative URL, '/players/B/BradTo00.htm'. Similarly,
    '201409070dal' refers to the boxscore of the SF @ DAL game on 09/07/14.

    Supported types:
    * player/...
    * boxscores/...
    * teams/...
    * years/...
    * leagues/...
    * awards/...
    * coaches/...
    * officials/...
    * schools/...
    * schools/high_schools.cgi?id=...

    :returns: ID associated with the given relative URL.
    """
    yearRegex = r'.*/years/(\d{4}).*|.*/gamelog/(\d{4}).*'
    playerRegex = r'.*/players/(?:\w/)?(.+?)(?:/|\.html?)'
    boxscoresRegex = r'.*/boxscores/(.+?)\.html?'
    teamRegex = r'.*/teams/(\w{3})/.*'
    coachRegex = r'.*/coaches/(.+?)\.html?'
    stadiumRegex = r'.*/stadiums/(.+?)\.html?'
    refRegex = r'.*/officials/(.+?r)\.html?'
    collegeRegex = r'.*/schools/(\S+?)/.*|.*college=([^&]+)'
    hsRegex = r'.*/schools/high_schools\.cgi\?id=([^\&]{8})'
    bsDateRegex = r'.*/boxscores/index\.f?cgi\?(month=\d+&day=\d+&year=\d+)'
    leagueRegex = r'.*/leagues/(.*_\d{4}).*'
    awardRegex = r'.*/awards/(.+)\.htm'

    regexes = [
        yearRegex,
        playerRegex,
        boxscoresRegex,
        teamRegex,
        coachRegex,
        stadiumRegex,
        refRegex,
        collegeRegex,
        hsRegex,
        bsDateRegex,
        leagueRegex,
        awardRegex,
    ]

    for regex in regexes:
        match = re.match(regex, url, re.I)
        if match:
            return [_f for _f in match.groups() if _f][0]

    # things we don't want to match but don't want to print a WARNING
    if any(
        url.startswith(s) for s in
        (
            '/play-index/',
        )
    ):
        return url

    print('WARNING. NO MATCH WAS FOUND FOR "{}"'.format(url))
    return url

def get_html(url):
    """Gets the HTML for the given URL using a GET request.

    :url: the absolute URL of the desired page.
    :returns: a string of HTML.
    """
    global last_request_time
    with throttle_process_lock:
        with throttle_thread_lock:
            # sleep until THROTTLE_DELAY secs have passed since last request
            wait_left = THROTTLE_DELAY - (time.time() - last_request_time.value)
            if wait_left > 0:
                time.sleep(wait_left)

            # make request
            response = requests.get(url)

            # update last request time for throttling
            last_request_time.value = time.time()

    # raise ValueError on 4xx status code, get rid of comments, and return
    if 400 <= response.status_code < 500:
        raise ValueError(
            'Status Code {} received fetching URL "{}"'
            .format(response.status_code, url)
        )
    html = response.text
    html = html.replace('<!--', '').replace('-->', '')

    return html

def _subpage_url(yr, page):
    return (BASE_URL +
            '/leagues/NBA_{}_{}.html'.format(yr, page))

def get_main_doc(yr):
    """Returns PyQuery object for the main season URL.
    :returns: PyQuery object.
    """
    url = (BASE_URL +
            '/leagues/NBA_{}.html'.format(yr))
    return pq(get_html(url))

def get_sub_doc(yr, subpage):
    """Returns PyQuery object for a given subpage URL.
    :subpage: The subpage of the season, e.g. 'per_game'.
    :returns: PyQuery object.
    """
    html = get_html(_subpage_url(yr, subpage))
    time.sleep(1)
    return pq(html)

def flatten_links(td, _recurse=False):
    """Flattens relative URLs within text of a table cell to IDs and returns
    the result.

    :td: the PyQuery object for the HTML to convert
    :returns: the string with the links flattened to IDs
    """

    # helper function to flatten individual strings/links
    def _flatten_node(c):
        if isinstance(c, basestring):
            return c.strip()
        elif 'href' in c.attrib:
            c_id = rel_url_to_id(c.attrib['href'])
            return c_id if c_id else c.text_content().strip()
        else:
            return flatten_links(pq(c), _recurse=True)

    # if there's no text, just return None
    if td is None or not td.text():
        return '' if _recurse else None

    td.remove('span.note')
    return ''.join(_flatten_node(c) for c in td.contents())

def parse_table(table, flatten=True, footer=False):
    """Parses a table from sports-reference sites into a pandas dataframe.

    :param table: the PyQuery object representing the HTML table
    :param flatten: if True, flattens relative URLs to IDs. otherwise, leaves
        all fields as text without cleaning.
    :param footer: If True, returns the summary/footer of the page. Recommended
        to use this with flatten=False. Defaults to False.
    :returns: pd.DataFrame
    """
    if not len(table):
        return pd.DataFrame()

    # get columns
    columns = [c.attrib['data-stat']
               for c in table('thead tr:not([class]) th[data-stat]')]

    # get data
    rows = list(table('tbody tr' if not footer else 'tfoot tr')
                .not_('.thead, .stat_total, .stat_average').items())
    data = [
        [flatten_links(td) if flatten else td.text()
         for td in row.items('th,td')]
        for row in rows
    ]

    # make DataFrame
    df = pd.DataFrame(data, columns=columns, dtype='float')

    # add has_class columns
    allClasses = set(
        cls
        for row in rows
        if row.attr['class']
        for cls in row.attr['class'].split()
    )
    for cls in allClasses:
        df['has_class_' + cls] = [
            bool(row.attr['class'] and
                 cls in row.attr['class'].split())
            for row in rows
        ]

    # cleaning the DataFrame

    df.drop(['ranker', 'Xxx', 'Yyy', 'Zzz'],
            axis=1, inplace=True, errors='ignore')

    # year_id -> year (as int)
    if 'year_id' in df.columns:
        df.rename(columns={'year_id': 'year'}, inplace=True)
        if flatten:
            df.year = df.year.fillna(method='ffill')
            df['year'] = df.year.map(lambda s: str(s)[:4]).astype(int)

    # pos -> position
    if 'pos' in df.columns:
        df.rename(columns={'pos': 'position'}, inplace=True)

    # boxscore_word, game_date -> boxscore_id and separate into Y, M, D columns
    for bs_id_col in ('boxscore_word', 'game_date', 'box_score_text','date_game'):
        if bs_id_col in df.columns:
            df.rename(columns={bs_id_col: 'boxscore_id'}, inplace=True)
            break

    # ignore *, +, and other characters used to note things
    df.replace(re.compile(r'[\*\+\u2605]', re.U), '', inplace=True)
    for col in df.columns:
        if hasattr(df[col], 'str'):
            df[col] = df[col].str.strip()

    # player -> player_id and/or player_name
    if 'player' in df.columns:
        if flatten:
            df.rename(columns={'player': 'player_id'}, inplace=True)
            # when flattening, keep a column for names
            player_names = parse_table(table, flatten=False)['player_name']
            df['player_name'] = player_names
        else:
            df.rename(columns={'player': 'player_name'}, inplace=True)

    # team, team_name -> team_id
    for team_col in ('team', 'team_name'):
        if team_col in df.columns:
            # first, get rid of faulty rows
            df = df.loc[~df[team_col].isin(['XXX'])]
            if flatten:
                df.rename(columns={team_col: 'team_id'}, inplace=True)

    # season -> int
    if 'season' in df.columns and flatten:
        df['season'] = df['season'].astype(int)

    # handle date_game columns (different types)
    if 'date_game' in df.columns and flatten:
        date_re = r'month=(?P<month>\d+)&day=(?P<day>\d+)&year=(?P<year>\d+)'
        date_df = df['date_game'].str.extract(date_re, expand=True)
        if date_df.notnull().all(axis=1).any():
            df = pd.concat((df, date_df), axis=1)
        else:
            df.rename(columns={'date_game': 'boxscore_id'}, inplace=True)

    # game_location -> is_home
    if 'game_location' in df.columns and flatten:
        df['game_location'] = df['game_location'].isnull()
        df.rename(columns={'game_location': 'is_home'}, inplace=True)

    # mp: (min:sec) -> float(min + sec / 60), notes -> NaN, new column
    if 'mp' in df.columns and df.dtypes['mp'] == object and flatten:
        mp_df = df['mp'].str.extract(
            r'(?P<m>\d+):(?P<s>\d+)', expand=True).astype(float)
        no_match = mp_df.isnull().all(axis=1)
        if no_match.any():
            df.loc[no_match, 'note'] = df.loc[no_match, 'mp']
        df['mp'] = mp_df['m'] + mp_df['s'] / 60

    # converts number-y things to floats
    def convert_to_float(val):
        # percentages: (number%) -> float(number * 0.01)
        m = re.search(r'([-\.\d]+)\%',
                      val if isinstance(val, basestring) else str(val), re.U)
        try:
            if m:
                return float(m.group(1)) / 100 if m else val
            if m:
                return int(m.group(1)) + int(m.group(2)) / 60
        except ValueError:
            return val
        # salaries: $ABC,DEF,GHI -> float(ABCDEFGHI)
        m = re.search(r'\$[\d,]+',
                      val if isinstance(val, basestring) else str(val), re.U)
        try:
            if m:
                return float(re.sub(r'\$|,', '', val))
        except Exception:
            return val
        # generally try to coerce to float, unless it's an int or bool
        try:
            if isinstance(val, (int, bool)):
                return val
            else:
                return float(val)
        except Exception:
            return val

    if flatten:
        df = df.applymap(convert_to_float)

    df = df.loc[df.astype(bool).any(axis=1)]

    return df

def return_schedule(year, month_nums):
    """Returns a list of BoxScore IDs for every game in the season.

    :returns: DataFrame of schedule information.
    :rtype: pd.DataFrame
    """
    # kind = kind.upper()[0]
    dfs = []

    if month_nums:
        month_nums = month_nums.split(',')
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
        months = (month_map[num] for num in month_nums)
    else:
        months=('october', 'november', 'december', 'january', 'february',
                    'march', 'april', 'may', 'june','july','august')

    # get games from each month
    for month in months:
        try:
            time.sleep(1)
            doc = get_sub_doc(year, 'games-{}'.format(month))
        except ValueError:
            continue
        table = doc('table#schedule')
        df = parse_table(table)
        if isinstance(df, pd.DataFrame):
            dfs.append(df)
        else:
            print(f'\n\n*** MONTH NOT INSERTED {month} ***\n\n')
        time.sleep(5)
    df = pd.concat(dfs).reset_index(drop=True)

    return df


## PBP Specific Functions

def get_subpage_doc(boxscore_id, page):
    url = (BASE_URL +
            '/boxscores/{}/{}.html'.format(page, boxscore_id))
    doc = pq(get_html(url))
    return doc

def get_game_doc(boxscore_id):
    url = ('{}/boxscores/{}.html'
            .format(BASE_URL, boxscore_id))
    doc = pq(get_html(url))
    return doc

def linescore(boxscore_id):
    """Returns the linescore for the game as a DataFrame."""
    doc = get_game_doc(boxscore_id)
    table = doc('table#line_score')

    columns = [th.text() for th in table('tr').items('th')][1:-2]
    columns[0] = 'team_id'

    data = [flatten_links(td) for td in table('td').items()]
    teams = [flatten_links(td) for td in table('a').items()]

    data = np.array(data).reshape(2,-1)
    teams = np.array(teams).reshape(2,-1)

    return pd.DataFrame(np.hstack((teams, data)), index=['away', 'home'],
                                columns=columns)  

def parse_play(boxscore_id, season, details, is_hm):
    """Parse play details from a play-by-play string describing a play.

    Assuming valid input, this function returns structured data in a dictionary
    describing the play. If the play detail string was invalid, this function
    returns None.

    :param boxscore_id: the boxscore ID of the play
    :param details: detail string for the play
    :param is_hm: bool indicating whether the offense is at home
    :param returns: dictionary of play attributes or None if invalid
    :rtype: dictionary or None
    """
    # if input isn't a string, return None
    if not details or not isinstance(details, basestring):
        return None

    _linescore = linescore(boxscore_id)
    hm = _linescore.loc['home', 'team_id']
    aw = _linescore.loc['away', 'team_id']
    hm_roster = set(_get_player_stats(boxscore_id).query('is_home == True').player_id.values)

    p = {}
    p['detail'] = details
    p['home'] = hm
    p['away'] = aw
    p['is_home_play'] = is_hm

    # parsing field goal attempts
    #shotRE = (r'(?P<shooter>{0}) (?P<is_fgm>makes|misses) '
    #          '(?P<is_three>2|3)\-pt shot').format(PLAYER_RE)
    #distRE = r' (?:from (?P<shot_dist>\d+) ft|at rim)'
    #assistRE = r' \(assist by (?P<assister>{0})\)'.format(PLAYER_RE)
    #blockRE = r' \(block by (?P<blocker>{0})\)'.format(PLAYER_RE)
    #shotRE = r'{0}{1}(?:{2}|{3})?'.format(shotRE, distRE, assistRE, blockRE)

    shotRE = (r'(?P<shooter>{0}) ?(?P<is_fgm>makes|misses) '
              '(?P<is_three>2|3)\-pt (driving layup|layup|shot|hook shot|jump shot|dunk|tip-in)').format(PLAYER_RE)
    distRE = r' (?:from (?P<shot_dist>\d+) ft|at rim)'
    assistRE = r' \(assist by ?(?P<assister>{0})\)'.format(PLAYER_RE)
    blockRE = r' \(block by ?(?P<blocker>{0})\)'.format(PLAYER_RE)
    shotRE = r'{0}{1}(?:{2}|{3})?'.format(shotRE, distRE, assistRE, blockRE)
    m = re.match(shotRE, details, re.IGNORECASE)
    if m:
        p['is_fga'] = True
        p.update(m.groupdict())
        p['shot_dist'] = p['shot_dist'] if p['shot_dist'] is not None else 0
        p['shot_dist'] = int(p['shot_dist'])
        p['is_fgm'] = p['is_fgm'] == 'makes'
        p['is_three'] = p['is_three'] == '3'
        p['is_assist'] = pd.notnull(p.get('assister'))
        p['is_block'] = pd.notnull(p.get('blocker'))
        shooter_home = p['shooter'] in hm_roster
        p['off_team'] = hm if shooter_home else aw
        p['def_team'] = aw if shooter_home else hm
        return p
    #'Jump ball:noahjo01vs.thomptr01(rosede01gains possession)'

    # parsing jump balls
    jumpRE = ((r'Jump ball: ?(?P<away_jumper>{0}) ?vs\. ?(?P<home_jumper>{0})'
               r'(?: ?\((?P<gains_poss>{0}) ?gains possession\))?')
              .format(PLAYER_RE))
    m = re.match(jumpRE, details, re.IGNORECASE)
    if m:
        p['is_jump_ball'] = True
        p.update(m.groupdict())
        return p

    # parsing rebounds
    rebRE = (r'(?P<is_oreb>Offensive|Defensive) rebound'
             r' by ?(?P<rebounder>{0}|Team)').format(PLAYER_RE)
    m = re.match(rebRE, details, re.I)
    if m:
        p['is_reb'] = True
        p.update(m.groupdict())
        p['is_oreb'] = p['is_oreb'].lower() == 'offensive'
        p['is_dreb'] = not p['is_oreb']
        if p['rebounder'] == 'Team':
            p['reb_team'], other = (hm, aw) if is_hm else (aw, hm)
        else:
            reb_home = p['rebounder'] in hm_roster
            p['reb_team'], other = (hm, aw) if reb_home else (aw, hm)
        p['off_team'] = p['reb_team'] if p['is_oreb'] else other
        p['def_team'] = p['reb_team'] if p['is_dreb'] else other
        return p

    # parsing free throws
    ftRE = (r'(?P<ft_shooter>{}) ?(?P<is_ftm>makes|misses) '
            r'(?P<is_tech_fta>technical )?(?P<is_flag_fta>flagrant )?'
            r'(?P<is_clearpath_fta>clear path )?free throw'
            r'(?: (?P<fta_num>\d+) of (?P<tot_fta>\d+))?').format(PLAYER_RE)
    m = re.match(ftRE, details, re.I)
    if m:
        p['is_fta'] = True
        p.update(m.groupdict())
        p['is_ftm'] = p['is_ftm'] == 'makes'
        p['is_tech_fta'] = bool(p['is_tech_fta'])
        p['is_flag_fta'] = bool(p['is_flag_fta'])
        p['is_clearpath_fta'] = bool(p['is_clearpath_fta'])
        p['is_pf_fta'] = not p['is_tech_fta']
        if p['tot_fta']:
            p['tot_fta'] = int(p['tot_fta'])
        if p['fta_num']:
            p['fta_num'] = int(p['fta_num'])
        ft_home = p['ft_shooter'] in hm_roster
        p['fta_team'] = hm if ft_home else aw
        if not p['is_tech_fta']:
            p['off_team'] = hm if ft_home else aw
            p['def_team'] = aw if ft_home else hm
        return p

    # parsing substitutions
    subRE = (r'(?P<sub_in>{0}) ?enters the game for ?'
             r'(?P<sub_out>{0})').format(PLAYER_RE)
    m = re.match(subRE, details, re.I)
    if m:
        p['is_sub'] = True
        p.update(m.groupdict())
        sub_home = p['sub_in'] in hm_roster or p['sub_out'] in hm_roster
        p['sub_team'] = hm if sub_home else aw
        return p

    # parsing turnovers
    toReasons = (r'(?P<to_type>[^;]+)(?:; steal by ?'
                 r'(?P<stealer>{0}))?').format(PLAYER_RE)
    toRE = (r'Turnover by ?(?P<to_by>{}|Team| ) ?'
            r'\((?:{})\)').format(PLAYER_RE, toReasons)
    m = re.match(toRE, details, re.I)
    if m:
        p['is_to'] = True
        p.update(m.groupdict())
        p['to_type'] = p['to_type'].lower()
        if p['to_type'] == 'offensive foul':
            return None
        p['is_steal'] = pd.notnull(p['stealer'])
        p['is_travel'] = p['to_type'] == 'traveling'
        p['is_shot_clock_viol'] = p['to_type'] == 'shot clock'
        p['is_oob'] = p['to_type'] == 'step out of bounds'
        p['is_three_sec_viol'] = p['to_type'] == '3 sec'
        p['is_backcourt_viol'] = p['to_type'] == 'back court'
        p['is_off_goaltend'] = p['to_type'] == 'offensive goaltending'
        p['is_double_dribble'] = p['to_type'] == 'dbl dribble'
        p['is_discont_dribble'] = p['to_type'] == 'discontinued dribble'
        p['is_carry'] = p['to_type'] == 'palming'
        if p['to_by'] == 'Team':
            p['off_team'] = hm if is_hm else aw
            p['def_team'] = aw if is_hm else hm
        else:
            to_home = p['to_by'] in hm_roster
            p['off_team'] = hm if to_home else aw
            p['def_team'] = aw if to_home else hm
        return p

    # parsing shooting fouls
    shotFoulRE = (r'Shooting(?P<is_block_foul> block)? foul by ?(?P<fouler>{0})'
                  r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(shotFoulRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_shot_foul'] = True
        p.update(m.groupdict())
        p['is_block_foul'] = bool(p['is_block_foul'])
        foul_on_home = p['fouler'] in hm_roster
        p['off_team'] = aw if foul_on_home else hm
        p['def_team'] = hm if foul_on_home else aw
        p['foul_team'] = p['def_team']
        return p

    # parsing offensive fouls
    offFoulRE = (r'Offensive(?P<is_charge> charge)? foul '
                 r'by ?(?P<to_by>{0})'
                 r'(?: \(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(offFoulRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_off_foul'] = True
        p['is_to'] = True
        p['to_type'] = 'offensive foul'
        p.update(m.groupdict())
        p['is_charge'] = bool(p['is_charge'])
        p['fouler'] = p['to_by']
        foul_on_home = p['fouler'] in hm_roster
        p['off_team'] = hm if foul_on_home else aw
        p['def_team'] = aw if foul_on_home else hm
        p['foul_team'] = p['off_team']
        return p

    # parsing personal fouls
    foulRE = (r'Personal (?P<is_take_foul>take )?(?P<is_block_foul>block )?'
              r'foul by ?(?P<fouler>{0})(?: ?\(drawn by ?'
              r'(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(foulRE, details, re.I)
    if m:
        p['is_pf'] = True
        p.update(m.groupdict())
        p['is_take_foul'] = bool(p['is_take_foul'])
        p['is_block_foul'] = bool(p['is_block_foul'])
        foul_on_home = p['fouler'] in hm_roster
        p['off_team'] = aw if foul_on_home else hm
        p['def_team'] = hm if foul_on_home else aw
        p['foul_team'] = p['def_team']
        return p

    # TODO: parsing double personal fouls
    # double_foul_re = (r'Double personal foul by (?P<fouler1>{0}) and '
    #                   r'(?P<fouler2>{0})').format(PLAYER_RE)
    # m = re.match(double_Foul_re, details, re.I)
    # if m:
    #     p['is_pf'] = True
    #     p.update(m.groupdict())
    #     p['off_team'] =

    # parsing loose ball fouls
    looseBallRE = (r'Loose ball foul by ?(?P<fouler>{0})'
                   r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(looseBallRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_loose_ball_foul'] = True
        p.update(m.groupdict())
        foul_home = p['fouler'] in hm_roster
        p['foul_team'] = hm if foul_home else aw
        return p

    # parsing punching fouls
    # TODO

    # parsing away from play fouls
    awayFromBallRE = ((r'Away from play foul by ?(?P<fouler>{0})'
                       r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?')
                      .format(PLAYER_RE))
    m = re.match(awayFromBallRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_away_from_play_foul'] = True
        p.update(m.groupdict())
        foul_on_home = p['fouler'] in hm_roster
        # TODO: figure out who had the ball based on previous play
        p['foul_team'] = hm if foul_on_home else aw
        return p

    # parsing inbound fouls
    inboundRE = (r'Inbound foul by ?(?P<fouler>{0})'
                 r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(inboundRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_inbound_foul'] = True
        p.update(m.groupdict())
        foul_on_home = p['fouler'] in hm_roster
        p['off_team'] = aw if foul_on_home else hm
        p['def_team'] = hm if foul_on_home else aw
        p['foul_team'] = p['def_team']
        return p

    # parsing flagrant fouls
    flagrantRE = (r'Flagrant foul type (?P<flag_type>1|2) by ?(?P<fouler>{0})'
                  r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(flagrantRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_flagrant'] = True
        p.update(m.groupdict())
        foul_on_home = p['fouler'] in hm_roster
        p['foul_team'] = hm if foul_on_home else aw
        return p

    # parsing clear path fouls
    clearPathRE = (r'Clear path foul by ?(?P<fouler>{0})'
                   r'(?: ?\(drawn by ?(?P<drew_foul>{0})\))?').format(PLAYER_RE)
    m = re.match(clearPathRE, details, re.I)
    if m:
        p['is_pf'] = True
        p['is_clear_path_foul'] = True
        p.update(m.groupdict())
        foul_on_home = p['fouler'] in hm_roster
        p['off_team'] = aw if foul_on_home else hm
        p['def_team'] = hm if foul_on_home else aw
        p['foul_team'] = p['def_team']
        return p

    # parsing timeouts
    timeoutRE = r'(?P<timeout_team>.*?) (?:full )?timeout'
    m = re.match(timeoutRE, details, re.I)
    if m:
        p['is_timeout'] = True
        p.update(m.groupdict())
        isOfficialTO = p['timeout_team'].lower() == 'official'
        name_to_id = season.team_names_to_ids()
        p['timeout_team'] = (
            'Official' if isOfficialTO else
            name_to_id.get(hm, name_to_id.get(aw, p['timeout_team']))
        )
        return p

    # parsing technical fouls
    techRE = (r'(?P<is_hanging>Hanging )?'
              r'(?P<is_taunting>Taunting )?'
              r'(?P<is_ill_def>Ill def )?'
              r'(?P<is_delay>Delay )?'
              r'(Technical)?'
              r'(?P<is_unsport>Non unsport )?'
              r'tech(?:nical)? foul by ?'
              r'(?P<tech_fouler>{0}|Team)').format(PLAYER_RE)
    ### May need to account for "Technical foul by(blank)"
    m = re.match(techRE, details, re.I)
    if m:
        p['is_tech_foul'] = True
        p.update(m.groupdict())
        p['is_hanging'] = bool(p['is_hanging'])
        p['is_taunting'] = bool(p['is_taunting'])
        p['is_ill_def'] = bool(p['is_ill_def'])
        p['is_delay'] = bool(p['is_delay'])
        p['is_unsport'] = bool(p['is_unsport'])
        foul_on_home = p['tech_fouler'] in hm_roster
        p['foul_team'] = hm if foul_on_home else aw
        return p

    # parsing ejections
    ejectRE = r'(?P<ejectee>{0}|Team) ?c?ejected from game'.format(PLAYER_RE)
    m = re.match(ejectRE, details, re.I)
    if m:
        p['is_ejection'] = True
        p.update(m.groupdict())
        if p['ejectee'] == 'Team':
            p['ejectee_team'] = hm if is_hm else aw
        else:
            eject_home = p['ejectee'] in hm_roster
            p['ejectee_team'] = hm if eject_home else aw
        return p

    # parsing defensive 3 seconds techs
    def3TechRE = (r'(?:Def 3 sec tech foul|Defensive three seconds)'
                  r' by ?(?P<tech_fouler>{}|Team)').format(PLAYER_RE)
    m = re.match(def3TechRE, details, re.I)
    if m:
        p['is_tech_foul'] = True
        p['is_def_three_secs'] = True
        p.update(m.groupdict())
        foul_on_home = p['tech_fouler'] in hm_roster
        p['off_team'] = aw if foul_on_home else hm
        p['def_team'] = hm if foul_on_home else aw
        p['foul_team'] = p['def_team']
        return p

    # parsing violations
    violRE = (r'Violation by ?(?P<violator>{0}|Team) ?'
              r'\((?P<viol_type>.*)\)').format(PLAYER_RE)
    m = re.match(violRE, details, re.I)
    if m:
        p['is_viol'] = True
        p.update(m.groupdict())
        if p['viol_type'] == 'kicked_ball':
            p['is_to'] = True
            p['to_by'] = p['violator']
        if p['violator'] == 'Team':
            p['viol_team'] = hm if is_hm else aw
        else:
            viol_home = p['violator'] in hm_roster
            p['viol_team'] = hm if viol_home else aw
        return p

    p['is_error'] = True
    return p

def sparse_lineup_cols(df):
    regex = '{}_in'.format(PLAYER_RE)
    return [c for c in df.columns if re.match(regex, c)]

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
            df[col] = (df[col] == True)

        # fill NaN's in sparse lineup columns to 0
        elif col in sparse_cols:
            df[col] = df[col].fillna(0)

    # fix free throw columns on technicals
    df.loc[df.is_tech_fta, ['fta_num', 'tot_fta']] = 1

    # fill in NaN's/fix off_team and def_team columns
    df.off_team.fillna(method='bfill', inplace=True)
    df.def_team.fillna(method='bfill', inplace=True)
    df.off_team.fillna(method='ffill', inplace=True)
    df.def_team.fillna(method='ffill', inplace=True)

    return df

def return_pbp(boxscore_id, season, dense_lineups=False, sparse_lineups=False):
    """Returns a dataframe of the play-by-play data from the game.

    :param dense_lineups: If True, adds 10 columns containing the names of
        the players on the court. Defaults to False.
    :param sparse_lineups: If True, adds binary columns denoting whether a
        given player is in the game at the time of a pass. Defaults to
        False.
    :returns: pandas DataFrame of play-by-play. Similar to GPF.
    """
    try:
        doc = get_subpage_doc(boxscore_id, 'pbp')
    except:
        raise ValueError(
            'Error fetching PBP subpage for boxscore {}'
            .format(boxscore_id)
        )
    table = doc('table#pbp')
    trs = [
        tr for tr in table('tr').items()
        if (not tr.attr['class'] or  # regular data rows
            tr.attr['id'] and tr.attr['id'].startswith('q'))  # qtr bounds
    ]
    rows = [tr.children('td') for tr in trs]
    n_rows = len(trs)
    data = []
    cur_qtr = 0
    bsid = boxscore_id

    for i in range(n_rows):
        try:
            tr = trs[i]
            row = rows[i]
            p = {}

            # increment cur_qtr when we hit a new quarter
            if tr.attr['id'] and tr.attr['id'].startswith('q'):
                assert int(tr.attr['id'][1:]) == cur_qtr + 1
                cur_qtr += 1
                continue

            # add time of play to entry
            t_str = row.eq(0).text()
            t_regex = r'(\d+):(\d+)\.(\d+)'
            mins, secs, tenths = map(int, re.match(t_regex, t_str).groups())
            endQ = (12 * 60 * min(cur_qtr, 4) +
                    5 * 60 * (cur_qtr - 4 if cur_qtr > 4 else 0))
            secsElapsed = endQ - (60 * mins + secs + 0.1 * tenths)
            p['secs_elapsed'] = secsElapsed
            p['clock_time'] = t_str
            p['quarter'] = cur_qtr

            # handle single play description
            # ex: beginning/end of quarter, jump ball
            if row.length == 2:
                desc = row.eq(1)
                # handle jump balls
                if desc.text().lower().startswith('jump ball: '):
                    p['is_jump_ball'] = True
                    jb_str = flatten_links(desc)
                    p.update(
                        parse_play(bsid, season, jb_str, None)
                    )
                # ignore rows marking beginning/end of quarters
                elif (
                    desc.text().lower().startswith('start of ') or
                    desc.text().lower().startswith('end of ')
                ):
                    continue
                # if another case, log and continue
                else:
                    if not desc.text().lower().startswith('end of '):
                        print(
                            '{}, Q{}, {} other case: {}'
                            .format(boxscore_id, cur_qtr,
                                    t_str, desc.text())
                        )
                    continue

            # handle team play description
            # ex: shot, turnover, rebound, foul, sub, etc.
            elif row.length == 6:
                aw_desc, hm_desc = row.eq(1), row.eq(5)
                is_hm_play = bool(hm_desc.text())
                desc = hm_desc if is_hm_play else aw_desc
                desc = flatten_links(desc)
                # parse the play
                new_p = parse_play(bsid, season, desc, is_hm_play)
                if not new_p:
                    continue
                elif isinstance(new_p, list):
                    # this happens when a row needs to be expanded to 2 rows;
                    # ex: double personal foul -> two PF rows

                    # first, update and append the first row
                    orig_p = dict(p)
                    p.update(new_p[0])
                    data.append(p)
                    # second, set up the second row to be appended below
                    p = orig_p
                    new_p = new_p[1]
                elif new_p.get('is_error'):
                    print("can't parse: {}, boxscore: {}"
                            .format(desc, boxscore_id))
                p.update(new_p)

            # otherwise, I don't know what this was
            else:
                raise Exception(("don't know how to handle row of length {}"
                                    .format(row.length)))

            data.append(p)
        except:
            print('Error parsing play-by-play for boxscore {}'
                    .format(boxscore_id))
            print('Current quarter: {}'.format(cur_qtr))
            print('Current row: {}'.format(i))
            print('Current row length: {}'.format(row.length))
            print('Current row: {}'.format(row))
            print('Current row text: {}'.format(row.text()))
            print('Current row class: {}'.format(tr.attr['class']))
            print('Current row id: {}'.format(tr.attr['id']))

    # convert to DataFrame and clean columns
    df = pd.DataFrame.from_records(data)
    df.sort_values('secs_elapsed', inplace=True, kind='mergesort')
    df = clean_features(df)

    # add columns for home team, away team, boxscore_id, date
    _linescore = linescore(boxscore_id)
    home = _linescore.loc['home', 'team_id']
    away = _linescore.loc['away', 'team_id']
    df['home'] = home
    df['away'] = away
    df['boxscore_id'] = boxscore_id
    df['season'] = season
    df['year'] = boxscore_id[:4]
    df['month'] = boxscore_id[4:6]
    df['day'] = boxscore_id[6:8]

    def _clean_rebs(df):
        df.reset_index(drop=True, inplace=True)
        no_reb_after = (
            (df.fta_num < df.tot_fta) | df.is_ftm |
            df.get('is_tech_fta', False)
        ).shift(1).fillna(False)
        no_reb_before = (
            (df.fta_num == df.tot_fta)
        ).shift(-1).fillna(False)
        se_end_qtr = df.loc[
            df.clock_time == '0:00.0', 'secs_elapsed'
        ].unique()
        no_reb_when = df.secs_elapsed.isin(se_end_qtr)
        drop_mask = (
            (df.rebounder == 'Team') &
            (no_reb_after | no_reb_before | no_reb_when)
        ).to_numpy().nonzero()[0]
        df.drop(drop_mask, axis=0, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # get rid of 'rebounds' after FTM, non-final FTA, or tech FTA
    df = _clean_rebs(df)

    # track possession number for each possession
    # FGM, dreb, TO, end of Q, made last FT, lost jump ball,
    # def goaltending, shot clock violation
    new_poss = (df.off_team == df.home).astype(int).diff().fillna(0)
    # def rebound considered part of the new possession
    #df['poss_id'] = np.cumsum(new_poss) + df.is_dreb.astype(int)
    #df['poss_id'] = np.abs(np.cumsum(new_poss) + df.is_dreb.astype(int))

    cum_sum_base = np.sum(np.cumsum(new_poss))/ np.abs(np.sum(np.cumsum(new_poss)))
    df['poss_id'] = cum_sum_base * np.where(df.is_dreb.astype(int) == 0, np.abs(np.cumsum(new_poss)), (1-np.abs(np.cumsum(new_poss))))

    # create poss_id with rebs -> new possessions for granular groupbys
    poss_id_reb = np.cumsum(new_poss.astype(int) | df.is_reb.astype(int))

    # make sure plays with the same clock time are in the right order
    sort_cols = [col for col in
                    ['is_reb', 'is_fga', 'is_pf', 'is_tech_foul',
                    'is_ejection', 'is_tech_fta', 'is_timeout', 'is_pf_fta',
                    'fta_num', 'is_viol', 'is_to', 'is_jump_ball', 'is_sub']
                    if col in df.columns]
    asc_true = ['fta_num']
    ascend = [(col in asc_true) for col in sort_cols]
    for label, group in df.groupby([df.secs_elapsed, poss_id_reb]):
        if len(group) > 1:
            df.loc[group.index, :] = group.sort_values(
                sort_cols, ascending=ascend, kind='mergesort'
            ).values

    # 2nd pass: get rid of 'rebounds' after FTM, non-final FTA, etc.
    df = _clean_rebs(df)

    # makes sure off/def and poss_id are correct for subs after rearranging
    # some possessions above
    df.loc[df['is_sub'], ['off_team', 'def_team', 'poss_id']] = np.nan
    df.off_team.fillna(method='bfill', inplace=True)
    df.def_team.fillna(method='bfill', inplace=True)
    df.poss_id.fillna(method='bfill', inplace=True)
    # make off_team and def_team NaN for jump balls
    if 'is_jump_ball' in df.columns:
        df.loc[df['is_jump_ball'], ['off_team', 'def_team']] = np.nan

    # make sure 'off_team' is always the team shooting FTs, even on techs
    # (impt for keeping track of the score)
    if 'is_tech_fta' in df.columns:
        tech_fta = df['is_tech_fta']
        df.loc[tech_fta, 'off_team'] = df.loc[tech_fta, 'fta_team']
        df.loc[tech_fta, 'def_team'] = np.where(
            df.loc[tech_fta, 'off_team'] == home, away, home
        )
    df.drop('fta_team', axis=1, inplace=True)
    # redefine poss_id_reb
    new_poss = (df.off_team == df.home).astype(int).diff().fillna(0)
    poss_id_reb = np.cumsum(new_poss.astype(int) | df.is_reb.astype(int))

    # get rid of redundant subs
    for (se, tm, pnum), group in df[df.is_sub].groupby(
        [df.secs_elapsed, df.sub_team, poss_id_reb]
    ):
        if len(group) > 1:
            sub_in = set()
            sub_out = set()
            # first, figure out who's in and who's out after subs
            for i, row in group.iterrows():
                if row['sub_in'] in sub_out:
                    sub_out.remove(row['sub_in'])
                else:
                    sub_in.add(row['sub_in'])
                if row['sub_out'] in sub_in:
                    sub_in.remove(row['sub_out'])
                else:
                    sub_out.add(row['sub_out'])
            assert len(sub_in) == len(sub_out)
            # second, add those subs
            n_subs = len(sub_in)
            for idx, p_in, p_out in zip(
                group.index[:n_subs], sub_in, sub_out
            ):
                assert df.loc[idx, 'is_sub']
                df.loc[idx, 'sub_in'] = p_in
                df.loc[idx, 'sub_out'] = p_out
                df.loc[idx, 'sub_team'] = tm
                df.loc[idx, 'detail'] = (
                    '{} enters the game for {}'.format(p_in, p_out)
                )
            # third, if applicable, remove old sub entries when there are
            # redundant subs
            n_extra = len(group) - len(sub_in)
            if n_extra:
                extra_idxs = group.index[-n_extra:]
                df.drop(extra_idxs, axis=0, inplace=True)

    df.reset_index(drop=True, inplace=True)

    # add column for pts and score
    df['pts'] = (df['is_ftm'] + 2 * df['is_fgm'] +
                    (df['is_fgm'] & df['is_three']))
    df['hm_pts'] = np.where(df.off_team == df.home, df.pts, 0)
    df['aw_pts'] = np.where(df.off_team == df.away, df.pts, 0)
    df['hm_score'] = np.cumsum(df['hm_pts'])
    df['aw_score'] = np.cumsum(df['aw_pts'])

    # more helpful columns
    # "play" is differentiated from "poss" by counting OReb as new play
    # "plays" end with non-and1 FGA, TO, last non-tech FTA, or end of qtr
    # (or double lane viol)
    new_qtr = df.quarter.diff().shift(-1).fillna(False).astype(bool)
    and1 = (df.is_fgm & df.is_pf.shift(-1).fillna(False) &
            df.is_fta.shift(-2).fillna(False) &
            ~df.secs_elapsed.diff().shift(-1).fillna(False).astype(bool))
    double_lane = (df.get('viol_type') == 'double lane')
    new_play = df.eval('(is_fga & ~(@and1)) | is_to | @new_qtr |'
                        '(is_fta & ~is_tech_fta & fta_num == tot_fta) |'
                        '@double_lane')
    df['play_id'] = np.cumsum(new_play).shift(1).fillna(0)
    df['hm_off'] = df.off_team == df.home

    # get lineup data
    if dense_lineups:
        df = pd.concat(
            (df, get_dense_lineups(df)), axis=1
        )
    if sparse_lineups:
        df = pd.concat(
            (df, get_sparse_lineups(df)), axis=1
        )

    return df

def _get_player_stats(boxscore_id, table_id_fmt='box-{}-game-basic'):
    """Returns a DataFrame of player stats from the game (either basic or
    advanced, depending on the argument.

    :param table_id_fmt: Format string for str.format with a placeholder
        for the team ID (e.g. 'box_{}_basic')
    :returns: DataFrame of player stats
    """

    # get data
    doc = get_game_doc(boxscore_id)
    _linescore = linescore(boxscore_id)
    hm = _linescore.loc['home', 'team_id']
    aw = _linescore.loc['away', 'team_id']   
    tms = aw, hm
    tm_ids = [table_id_fmt.format(tm) for tm in tms]
    tables = [doc('table#{}'.format(tm_id)) for tm_id in tm_ids]
    dfs = [parse_table(table) for table in tables]

    # clean data and add features
    for i, (tm, df) in enumerate(zip(tms, dfs)):
        no_time = df['mp'] == 0
        stat_cols = [col for col, dtype in df.dtypes.items()
                        if dtype != 'object']
        df.loc[no_time, stat_cols] = 0
        df['team_id'] = tm
        df['is_home'] = i == 1
        df['is_starter'] = [p < 5 for p in range(df.shape[0])]
        df.drop_duplicates(subset='player_id', keep='first', inplace=True)

    return pd.concat(dfs)

def get_period_starters(df):

    def players_from_play(play):
        """Figures out what players are in the game based on the players
        mentioned in a play. Returns away and home players as two sets.

        :param play: A dictionary representing a parsed play.
        :returns: (aw_players, hm_players)
        :rtype: tuple of lists
        """
        # if it's a tech FT from between periods, don't count this play
        if (
            play['clock_time'] == '12:00.0' and
            (play.get('is_tech_foul') or play.get('is_tech_fta'))
        ):
            return [], []

        stats = _get_player_stats(play['boxscore_id'])
        home_grouped = stats.groupby('is_home')
        hm_roster = set(home_grouped.player_id.get_group(True).values)
        aw_roster = set(home_grouped.player_id.get_group(False).values)
        player_keys = [
            'assister', 'away_jumper', 'blocker', 'drew_foul', 'fouler',
            'ft_shooter', 'gains_poss', 'home_jumper', 'rebounder', 'shooter',
            'stealer', 'sub_in', 'sub_out', 'to_by'
        ]
        players = [p for p in play[player_keys] if pd.notnull(p)]

        aw_players = [p for p in players if p in aw_roster]
        hm_players = [p for p in players if p in hm_roster]
        return aw_players, hm_players

    # create a mapping { quarter => (away_starters, home_starters) }
    n_periods = df.quarter.nunique()
    period_starters = [(set(), set()) for _ in range(n_periods)]

    # fill out this mapping quarter by quarter
    for qtr, qtr_grp in df.groupby(df.quarter):
        aw_starters, hm_starters = period_starters[qtr-1]
        exclude = set()
        # loop through sets of plays that happen at the "same time"
        for label, time_grp in qtr_grp.groupby(qtr_grp.secs_elapsed):
            # first, if they sub in and weren't already starters, exclude them
            sub_ins = set(time_grp.sub_in.dropna().values)
            exclude.update(sub_ins - aw_starters - hm_starters)
            # second, figure out new starters from each play at this time
            for i, row in time_grp.iterrows():
                aw_players, hm_players = players_from_play(row)
                # update overall sets for the quarter
                aw_starters.update(aw_players)
                hm_starters.update(hm_players)
            # remove excluded (subbed-in) players
            hm_starters -= exclude
            aw_starters -= exclude
            # check whether we have found all starters
            if len(hm_starters) >= 5 and len(aw_starters) >= 5:
                break

        if len(hm_starters) != 5 or len(aw_starters) != 5:
            print('WARNING: wrong number of starters for a team in Q{} of {}'
                  .format(qtr, df.boxscore_id.iloc[0]))

    return period_starters

def get_sparse_lineups(df):
    # get the lineup data using get_dense_lineups if necessary
    if (set(ALL_LINEUP_COLS) - set(df.columns)):
        lineup_df = get_dense_lineups(df)
    else:
        lineup_df = df[ALL_LINEUP_COLS]

    # create the sparse representation
    hm_lineups = lineup_df[HM_LINEUP_COLS].values
    aw_lineups = lineup_df[AW_LINEUP_COLS].values
    # +1 for home, -1 for away
    hm_df = pd.DataFrame([
        {'{}_in'.format(player_id): 1 for player_id in lineup}
        for lineup in hm_lineups
    ], dtype=int)
    aw_df = pd.DataFrame([
        {'{}_in'.format(player_id): -1 for player_id in lineup}
        for lineup in aw_lineups
    ], dtype=int)
    sparse_df = pd.concat((hm_df, aw_df), axis=1).fillna(0)
    return sparse_df


def get_dense_lineups(df):
    """Returns a new DataFrame based on the one it is passed. Specifically, it
    adds five columns for each team (ten total), where each column has the ID
    of a player on the court during the play.

    This information is figured out sequentially from the game's substitution
    data in the passed DataFrame, so the DataFrame passed as an argument must
    be from a specific BoxScore (rather than a DataFrame of non-consecutive
    plays). That is, the DataFrame must be of the form returned by
    :func:`nba.BoxScore.pbp <nba.BoxScore.pbp>`.

    .. note:: Note that the lineups reflect the teams in the game when the play
        happened, not after the play. For example, if a play is a substitution,
        the lineups for that play will be the lineups before the substituion
        occurs.

    :param df: A DataFrame of a game's play-by-play data.
    :returns: A DataFrame with additional lineup columns.

    """
    # TODO: add this precondition to documentation
    assert df['boxscore_id'].nunique() == 1

    def lineup_dict(aw_lineup, hm_lineup):
        """Returns a dictionary of lineups to be converted to columns.
        Specifically, the columns are 'aw_player1' through 'aw_player5' and
        'hm_player1' through 'hm_player5'.

        :param aw_lineup: The away team's current lineup.
        :param hm_lineup: The home team's current lineup.
        :returns: A dictionary of lineups.
        """
        return {
            '{}_player{}'.format(tm, i+1): player
            for tm, lineup in zip(['aw', 'hm'], [aw_lineup, hm_lineup])
            for i, player in enumerate(lineup)
        }

    def handle_sub(row, aw_lineup, hm_lineup):
        """Modifies the aw_lineup and hm_lineup lists based on the substitution
        that takes place in the given row."""
        assert row['is_sub']
        sub_lineup = hm_lineup if row['sub_team'] == row['home'] else aw_lineup
        try:
            # make the sub
            idx = sub_lineup.index(row['sub_out'])
            sub_lineup[idx] = row['sub_in']
        except ValueError:
            # if the sub was double-entered and it's already been executed...
            if (
                row['sub_in'] in sub_lineup
                and row['sub_out'] not in sub_lineup
            ):
                return aw_lineup, hm_lineup
            # otherwise, let's print and pretend this never happened
            print('ERROR IN SUB IN {}, Q{}, {}: {}'
                  .format(row['boxscore_id'], row['quarter'],
                          row['clock_time'], row['detail']))
            raise
        return aw_lineup, hm_lineup

    per_starters = get_period_starters(df)
    cur_qtr = 0
    aw_lineup, hm_lineup = [], []
    df = df.reset_index(drop=True)
    lineups = [{} for _ in range(df.shape[0])]

    # loop through select plays to determine lineups
    sub_or_per_start = df.is_sub | df.quarter.diff().astype(bool)
    for i, row in df.loc[sub_or_per_start].iterrows():
        if row['quarter'] > cur_qtr:
            # first row in a quarter
            assert row['quarter'] == cur_qtr + 1
            # first, finish up the last quarter's lineups
            if cur_qtr > 0 and not df.loc[i-1, 'is_sub']:
                lineups[i-1] = lineup_dict(aw_lineup, hm_lineup)
            # then, move on to the quarter, and enter the starting lineups
            cur_qtr += 1
            aw_lineup, hm_lineup = map(list, per_starters[cur_qtr-1])
            lineups[i] = lineup_dict(aw_lineup, hm_lineup)
            # if the first play in the quarter is a sub, handle that
            if row['is_sub']:
                aw_lineup, hm_lineup = handle_sub(row, aw_lineup, hm_lineup)
        else:
            # during the quarter
            # update lineups first then change lineups based on subs
            lineups[i] = lineup_dict(aw_lineup, hm_lineup)
            if row['is_sub']:
                aw_lineup, hm_lineup = handle_sub(row, aw_lineup, hm_lineup)

    # create and clean DataFrame
    lineup_df = pd.DataFrame(lineups)
    if lineup_df.iloc[-1].isnull().all():
        lineup_df.iloc[-1] = lineup_dict(aw_lineup, hm_lineup)
    lineup_df = lineup_df.groupby(df.quarter).fillna(method='bfill')

    # fill in NaN's based on minutes played
    bool_mat = lineup_df.isnull()
    mask = bool_mat.any(axis=1)
    if mask.any():
        # first, get the true minutes played from the box score
        stats = _get_player_stats(df.boxscore_id.iloc[0])
        true_mp = pd.Series(
            stats.query('mp > 0')[['player_id', 'mp']]
            .set_index('player_id').to_dict()['mp']
        ) * 60
        # next, calculate minutes played based on the lineup data
        calc_mp = pd.Series(
            {p: (df.secs_elapsed.diff() *
                 [p in row for row in lineup_df.values]).sum()
             for p in stats.query('mp > 0').player_id.values})
        # finally, figure which players are missing minutes
        diff = true_mp - calc_mp
        players_missing = diff.loc[diff.abs() >= 150]
        hm_roster = stats.query('is_home == True').player_id.values
        missing_df = pd.DataFrame(
            {'secs': players_missing.values,
             'is_home': players_missing.index.isin(hm_roster)},
            index=players_missing.index
        )

        if missing_df.empty:
            print('There are NaNs in the lineup data, but no players were '
                  'found to be missing significant minutes')
        else:
            for is_home, group in missing_df.groupby('is_home'):
                player_id = group.index.item()
                tm_cols = (HM_LINEUP_COLS if is_home else
                           AW_LINEUP_COLS)
                row_mask = lineup_df[tm_cols].isnull().any(axis=1)
                lineup_df.loc[row_mask, tm_cols] = (
                    lineup_df.loc[row_mask, tm_cols].fillna(player_id).values
                )

    return lineup_df

def clean_multigame_features(df):

    df = pd.DataFrame(df)
    if df.index.value_counts().max() > 1:
        df.reset_index(drop=True, inplace=True)

    df = clean_features(df)

    # if it's many games in one DataFrame, make poss_id and play_id unique
    for col in ('play_id', 'poss_id'):
        df[col] = df[col].fillna(method='bfill')
        df[col] = df[col].fillna(method='ffill')
        diffs = df[col].astype(int).diff().fillna(0)
        if (diffs < 0).any():
            new_col = np.cumsum(diffs.astype(bool))
            df.eval('{} = @new_col'.format(col), inplace=True)

    return df
