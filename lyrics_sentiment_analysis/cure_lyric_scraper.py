import re
import time

import polars as pl
import requests
from bs4 import BeautifulSoup

URL = "https://www.songlyrics.com/the-cure/"
DATA_DIR = "~/lyric-sentiment-analysis/data/"
DELAY_IN_SECONDS = 2.0


def get_lyrics(df: pl.DataFrame) -> pl.DataFrame:
    """
    Pulls the song lyrics from website and add as column in DataFrame
    :param df: DataFrame containing band, song names, URLs from which to pull lyrics
    :return: The dataframe including a column containing the song lyrics
    """
    lyrics_list = []
    for i, lyrics_url in enumerate(df["lyrics_url"].to_list()):
        song_title = df["Song Title"][i]
        print(f"Song {i} of {len(df)}: '{song_title}'")
        page = requests.get(lyrics_url)
        soup = BeautifulSoup(page.content, "html.parser")
        results = soup.find(id="songLyricsDiv")
        song_lyrics = results.text.replace("\n", " ").replace("\t", "").replace(
            "\r", "")
        lyrics_list.append(song_lyrics)
        time.sleep(DELAY_IN_SECONDS)

    df = df.with_columns(pl.Series(name="lyrics", values=lyrics_list))

    return df


def remove_extra_chars(txt: str) -> str:
    """
    This function removes unwanted extra characters from a string

    :param txt: Text string
    :return: Text string with extra characters removed
    """
    chars = re.compile("[':&"
                       "]+", re.UNICODE)
    return re.sub(chars, ' ', txt)


def add_dashes(txt: str) -> str:
    """
    This function replaces spaces with dashes in a string

    :param txt: Text string
    :return: Text string with spaces replaced with dashes
    """
    chars = re.compile(" ", re.UNICODE)
    return re.sub(chars, '-', txt)


if __name__ == "__main__":
    df_songs = pl.read_csv(DATA_DIR + "cure_songs.csv")

    df_songs = df_songs.with_columns(pl.col("Song Title")
                                     .map_elements(lambda x: remove_extra_chars(x),
                                                   return_dtype=pl.String).alias("Song Title Updated")
                                     )
    df_songs = df_songs.with_columns(pl.col("Song Title Updated")
                                     .map_elements(lambda x: add_dashes(x),
                                                   return_dtype=pl.String)
                                     )
    df_songs = df_songs.with_columns(pl.col("Song Title Updated").str.to_lowercase())

    df_songs = df_songs.with_columns(
        (URL + pl.col("Song Title Updated") + "-lyrics/").alias(
            "lyrics_url"))

    df_songs = get_lyrics(df_songs)

    df_songs.write_csv(DATA_DIR + "cure_song_lyrics.csv")
