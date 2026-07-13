BASE_CONFIG = {
    'train_neg_sample_args': None,
    'epochs': 20,
    'MAX_ITEM_LIST_LENGTH': 10,
    'eval_args': {
        'mode': {
            'valid': 'uni100',
            'test': 'uni100'
        },
        'order': 'TO',
        'split': {
            'LS': 'valid_and_test'
        }
    },
    'show_progress': False,
}

DATASETS_CONFIG = [
    # clicks
    {
        "name": "diginetica",
        "session_key": "session_id",
        "item_key": "item_id",
        "time_key": "timestamp",
    },
    # clicks
    {
        "name": "lastfm",
        "session_key": "user_id",
        "item_key": "artist_id",
        "time_key": "timestamp",
    },
    # views
    {
        "name": "retailrocket",
        "session_key": "visitor_id",
        "item_key": "item_id",
        "time_key": "timestamp",
    },
    # buys
    {
        "name": "yoochoose",
        "session_key": "session_id",
        "item_key": "item_id",
        "time_key": "timestamp",
    },
]