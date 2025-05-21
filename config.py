# Python environment path - specify the exact path to your Python interpreter
PYTHON_ENV_PATH = "/usr/bin/python3"

# Season ending year
NBA_SEASON = 2025

# MongoDB configuration
MONGODB_CONFIG = {
    "uri": "mongodb://localhost:27017/",
    "database": "funaki_nba",
    "collections": {
        "pbp": "play_by_play",
        "schedule": "schedule",
        "summary": "summary",
    },
    "options": {
        "maxPoolSize": 50,
        "connect": True,
        "serverSelectionTimeoutMS": 5000
    }
}