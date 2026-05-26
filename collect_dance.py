import requests
import pandas as pd
import time
import sys
import re

TOKEN = sys.argv[1] if len(sys.argv) > 1 else "49ntSHDdx8Djwt3_1afiWS3p-8_Y9eU7uHgo44qGcBk-VsfBcRFHWYDoYmQJtQKT"

HEADERS = {"Authorization": f"Bearer {TOKEN}"}
BASE = "https://api.genius.com"

DANCE_ARTISTS = [
    "Ed Sheeran", "Marshmello", "Charlie Puth",
    "DJ Snake", "Ariana Grande", "Shawn Mendes", "French Montana",
    "Dua Lipa", "Nicki Minaj", "Harry Styles", "Justin Bieber",
    "Selena Gomez", "Sia", "Taylor Swift", "The Weeknd",
    "Doja Cat", "Katy Perry", "Camila Cabello", "BLACKPINK",
    "Michael Jackson", "Lady Gaga"
]

def search_artist_id(name):
    try:
        r = requests.get(f"{BASE}/search", headers=HEADERS, params={"q": name}, timeout=10)
        hits = r.json().get("response", {}).get("hits", [])
        for hit in hits:
            artist = hit["result"]["primary_artist"]
            if name.lower() in artist["name"].lower():
                return artist["id"], artist["name"]
    except Exception as e:
        print(f"Search error: {e}")
    return None, None

def get_artist_songs(artist_id, per_page=10):
    try:
        r = requests.get(f"{BASE}/artists/{artist_id}/songs",
                         headers=HEADERS,
                         params={"per_page": per_page, "sort": "popularity"},
                         timeout=10)
        return r.json().get("response", {}).get("songs", [])
    except:
        return []

def clean_html(text):
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#x27;', "'", text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n+', ' ', text)
    return text.strip()

def get_lyrics_from_url(url):
    try:
        hdrs = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        r = requests.get(url, headers=hdrs, timeout=15)
        html = r.text

        all_lyrics_parts = []

        containers = re.finditer(r'data-lyrics-container="true"[^>]*>(.*?)</div>\s*</div>', html, re.DOTALL)
        for container in containers:
            content = container.group(1)

            if content.count('<button') > 0 or content.count('<svg') > 0:
                continue

            cleaned = clean_html(content)
            if len(cleaned) > 20:
                all_lyrics_parts.append(cleaned)

        if all_lyrics_parts:
            lyrics = " ".join(all_lyrics_parts)
            if len(lyrics) > 100:
                return lyrics

        raw_containers = re.findall(r'data-lyrics-container="true"[^>]*>(.*?)</div>', html, re.DOTALL)
        combined = " ".join(raw_containers)

        # Extract text that follows <br> tags
        lines = re.split(r'<br\s*/?>', combined)
        lyric_lines = []
        for line in lines:
            cleaned = re.sub(r'<[^>]+>', '', line)
            cleaned = re.sub(r'&[a-z]+;', '', cleaned)
            cleaned = cleaned.strip()
            # Only keep lines look like lyrics
            if len(cleaned) > 2 and not cleaned.startswith('{'):
                lyric_lines.append(cleaned)

        if lyric_lines:
            lyrics = " ".join(lyric_lines)
            if len(lyrics) > 100:
                return lyrics

        return ""

    except Exception as e:
        print(f"Lyrics fetch error: {e}")
        return ""

songs_collected = []
TARGET = 120

for artist_name in DANCE_ARTISTS:
    if len(songs_collected) >= TARGET:
        break
    print(f"\nSearching: {artist_name} ({len(songs_collected)}/{TARGET})")
    artist_id, found_name = search_artist_id(artist_name)
    if not artist_id:
        print(f"Could not find artist")
        continue

    songs = get_artist_songs(artist_id, per_page=10)
    time.sleep(0.5)

    for song in songs:
        if len(songs_collected) >= TARGET:
            break
        title = song.get("title", "")
        url   = song.get("url", "")
        rd    = song.get("release_date_components")
        year  = str(rd["year"]) if rd and rd.get("year") else "2015"

        print(f"Fetching: {title} ({year})")
        lyrics = get_lyrics_from_url(url)
        time.sleep(1.5)

        if not lyrics or len(lyrics) < 100:
            print(f"Skipping - only got {len(lyrics) if lyrics else 0} chars")
            continue

        lyrics = " ".join(lyrics.split())
        songs_collected.append({
            "artist_name":  found_name,
            "track_name":   title,
            "release_date": year,
            "genre":        "Dance",
            "lyrics":       lyrics
        })
        print(f"({len(songs_collected)}/100) - {len(lyrics)} chars")


if len(songs_collected) == 0:
    print("ERROR: No songs collected. Exiting.")
    sys.exit(1)

df = pd.DataFrame(songs_collected[:120])
df = df[["artist_name", "track_name", "release_date", "genre", "lyrics"]]
df.to_csv("Student_dataset.csv", index=False)
print(df[["artist_name", "track_name", "release_date"]].to_string())