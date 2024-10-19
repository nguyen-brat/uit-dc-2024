import re

def clean_hashtag(caption):
    return re.sub(r'\n?#\S+', '', caption).strip()