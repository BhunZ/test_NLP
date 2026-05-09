import os

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

OUTPUT_FILE = "assets/transcripts.jsonl"
SUCCESS_CACHE_FILE = "assets/processed_success.txt"
FAILED_CACHE_FILE = "assets/processed_failed.txt"

MIN_SEGMENTS = 30

TARGET_MIN = 100
TARGET_MAX = 600

MAX_RETRIES = 3
RETRY_DELAY = 5

SLEEP_BETWEEN_CALLS_MIN = 5.0
SLEEP_BETWEEN_CALLS_MAX = 10.0
LONG_BREAK_EVERY_N_SUCCESS = 5
LONG_BREAK_SECONDS = 45

BLOCK_STREAK_THRESHOLD = 5
BLOCK_COOLDOWN_SECONDS = 900
HTTP_TIMEOUT = 30


PLAYLISTS = [
    {"name": "CME295_Transformers_LLMs", "url": "https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy"},
    {"name": "CS224R_RL", "url": "https://www.youtube.com/watch?v=EvHRQhMX7_w&list=PLoROMvodv4rPwxE0ONYRa_itZFdaKCylL"},
    {"name": "CME296_Diffusion", "url": "https://www.youtube.com/watch?v=agN3AlfGFrk&list=PLoROMvodv4rObv1FMizXqumgVVdzX4_05"},
    {"name": "CS224N_NLP", "url": "https://www.youtube.com/watch?v=rmVRLeJRkl4&list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4"},
    {"name": "CS25_Transformers", "url": "https://www.youtube.com/watch?v=P127jhj-8-Y&list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM"},
    {"name": "CS224U", "url": "https://www.youtube.com/watch?v=rha64cQRLs8&list=PLoROMvodv4rPt5D0zs3YhbWSZA8Q_DyiJ"},    
    {"name": "CS229_ML", "url": "https://www.youtube.com/watch?v=Bl4Feh_Mjvo&list=PLoROMvodv4rNyWOpJg_Yh4NSqI4Z4vOYy"},
    {"name": "CS336_GPU_TPU", "url": "https://www.youtube.com/watch?v=izZba4UA7iY&list=PLoROMvodv4rPgrvmYbBrxZCK_GwXvDVL3"},
    #{"name": "CS224W_GraphML", "url": "https://www.youtube.com/watch?v=JAB_plj2rbA&list=PLoROMvodv4rPLKxIpqhjhPgdQy7imNkDn"}

]
