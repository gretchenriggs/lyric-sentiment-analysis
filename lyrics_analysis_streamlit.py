"""
This script runs the streamlit app used to choose which song lyrics to analyze to get their sentiment
"""
# Basic imports
import sqlite3
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import streamlit as st
from ds_helpers import make_logger

import lyrics_sentiment_analysis.database as db
from lyrics_sentiment_analysis.constants import DATA_DIR, LYRICS_ANALYSIS_DBNAME
from lyrics_sentiment_analysis.get_sentiment import get_sentiment, get_album_sentiment, grade_sentiment

logger = make_logger(__name__)


def song_analysis(df: pl.DataFrame, database=LYRICS_ANALYSIS_DBNAME) -> pl.DataFrame:
    """
    Get the sentiment analysis for songs contained in the input dataframe

    :param df: DataFrame with column of songs to get sentiment analysis for
    :param database: Database containing song lyrics information
    :return: DataFrame containing additional column of sentiment for the songs
    """
    start_time = time.time()

    db.create_sqlite_database(database)
    db.create_lyrics_analysis_table(database)
    with sqlite3.connect(database) as conn:
        # Iterating through songs
        sentiment_list = []
        grade_list = []
        for i, lyrics in enumerate(df["lyrics"]):
            band = df["band_name"][i]
            song = df["Song Title"][i]
            logger.info(f"Song {i + 1} of {len(df)}: '{song}'")
            logger.info(f"Checking if song title already in {database} lyrics analysis table")
            song_in_table = db.song_in_table(conn, band, song)
            if song_in_table:
                # If the song title is already in the database table, retrieve it
                logger.info("Song title already in database. Retrieving sentiment now...")
                sentiment, grade = db.retrieve_song_sentiment(conn, band, song)
            else:
                sentiment = get_sentiment(lyrics)
                grade = grade_sentiment(sentiment)

                # Creating input to SQL table with appropriate format, making sure to update string fields to not include
                #     quotation marks that may have trouble in SQL
                lyrics_data_for_db = (db.quote_updater(band),
                                      db.quote_updater(song),
                                      db.quote_updater(df["Album"][i]),
                                      df["Year Recorded"][i],
                                      db.quote_updater(df["lyrics"][i]),
                                      db.quote_updater(sentiment),
                                      grade)
                logger.info("Inserting songs and sentiments into table")
                db.add_lyric_analysis(conn, lyrics_data_for_db)
                logger.info(f"Created a entry in lyrics_analysis table for band, song title: {band}, {song}")
            sentiment_list.append(sentiment)
            grade_list.append(grade)
    conn.close()

    df = df.with_columns(pl.Series(name="sentiment", values=sentiment_list),
                         pl.Series(name="sentiment_grade", values=grade_list))
    logger.info(f"RunTime to get song sentiment: {time.time() - start_time:.2f} sec")

    return df


def album_analysis(df: pl.DataFrame) -> tuple[str, float]:
    """
    Get a summary of the sentiment of the entire album

    :param df: DataFrame with column of song sentiments
    :return: String response for the summary of the entire album of songs & floating point average sentiment grade
    """
    sentiments = ""
    for sent in df["sentiment"]:
        sentiments = sentiments + sent
    album_sentiment = get_album_sentiment(sentiments)
    avg_sentiment = round(df_songs["sentiment_grade"].mean(), 1)
    return album_sentiment, avg_sentiment


def sentiment_grading(df: pl.DataFrame) -> pl.DataFrame:
    """
    Get the sentiment grade for songs contained in the input dataframe

    :param df: DataFrame with column of song sentiments to get a sentiment grade
    :return: DataFrame containing additional column of sentiment grade
    """
    start_time = time.time()
    # Iterating through prompts
    response_list = []
    for i, sentiment in enumerate(df["sentiment"].to_list()):
        song_title = df["Song Title"][i]
        logger.info(f"Song {i + 1} of {len(df)}: '{song_title}'")
        sentiment_grade = grade_sentiment(sentiment)
        response_list.append(sentiment_grade)
    df = df.with_columns(pl.Series(name="sentiment", values=response_list))
    st.write(f"RunTime to get song sentiment: {time.time() - start_time:.2f} sec")
    return df


if __name__ == "__main__":
    st.title("Lyrics Sentiment Analysis App")

    dtypes = {
        "Song Title": pl.String,
        "Album": pl.String,
        "Year Recorded": pl.String,
        "band_name": pl.String,
        "Song Title Updated": pl.String,
        "lyrics_url": pl.String,
        "lyrics": pl.String
    }

    df_songs = pl.read_csv(DATA_DIR + "cure_song_lyrics.csv", schema_overrides=dtypes)
    band_list = df_songs["band_name"].unique().sort()
    album_list = df_songs["Album"].unique().sort()
    year_list = df_songs["Year Recorded"].unique().sort()
    song_list = df_songs["Song Title"].unique().sort()

    with st.sidebar:
        band_name = st.selectbox("Band Name", df_songs["band_name"].unique().sort())
        filter_type = st.selectbox("Filter Type", ["Album Filter", "Song Filter", "Year Filter"])
        if filter_type == "Album Filter":
            album_name = st.selectbox("Album", df_songs["Album"].unique().sort())
            entire_album_analysis = st.checkbox("Add Entire Album Sentiment Analysis")
            df_songs = df_songs.filter(pl.col("Album") == album_name)
        elif filter_type == "Song Filter":
            song_name = st.selectbox("Song Title", df_songs["Song Title"].unique().sort())
            df_songs = df_songs.filter(pl.col("Song Title") == song_name).head(1)
        elif filter_type == "Year Filter":
            year_recorded = st.selectbox("Album Year", df_songs["Year Recorded"].unique().sort())
            df_songs = df_songs.filter(pl.col("Year Recorded") == year_recorded)
        else:
            st.write("Please choose which type of Filter to apply.")
        show_lyrics = st.checkbox("Show Song Lyrics")

    st.header("Song Sentiment Analysis", divider="gray")
    df_songs = song_analysis(df_songs)
    for i, song in enumerate(df_songs["Song Title"]):
        band = df_songs["band_name"][i]
        lyrics = df_songs["lyrics"][i]
        sentiment = df_songs["sentiment"][i]
        sentiment_grade = df_songs["sentiment_grade"][i]
        st.write(f"**{band} - {song}**")
        if show_lyrics:
            st.write(f"    Lyrics: {lyrics}")
        st.write(f"    {sentiment_grade}")
        st.write(f"    {sentiment}")

    if filter_type == "Album Filter":
        grades = df_songs.sort("Track")["sentiment_grade"]
        grades = np.array(grades.to_list()).reshape(1, -1)
        fig = plt.figure()
        plt.imshow(grades, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        plt.colorbar(label='Sentiment Grade')
        plt.title("Sentiment Grade by Track")
        st.pyplot(fig)
    elif filter_type == "Year Filter":
        grades = np.array(df_songs["sentiment_grade"].to_list()).reshape(1, -1)
        fig = plt.figure()
        plt.imshow(grades, cmap='coolwarm', vmin=-1.0, vmax=1.0)
        plt.colorbar(label='Sentiment Grade')
        plt.title("Sentiment Grade")
        st.pyplot(fig)

    if (filter_type == "Album Filter") and entire_album_analysis:
        entire_album_sentiment, mean_sentiment = album_analysis(df_songs)
        st.write(f"Album Sentiment: {entire_album_sentiment}")
        st.write(f"Album Sentiment Score: {mean_sentiment}")
