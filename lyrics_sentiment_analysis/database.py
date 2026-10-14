"""
Functions for creating and querying database
"""

import codecs
import sqlite3
from typing import Union

LYRICS_ANALYSIS_TABLE_SCHEMA = {"band": str, "song": str, "album": str, "year_recorded": int, "lyrics": str,
                                "sentiment": str, "sentiment_grade": float}


def create_sqlite_database(filename: str) -> None:
    """
    Create a database connection to an SQLite database
    :param filename: SQL Lite database file name
    :return: None
    """
    conn = None
    try:
        conn = sqlite3.connect(filename)
        print(sqlite3.sqlite_version)
    except sqlite3.Error as e:
        print(e)
    finally:
        if conn:
            conn.close()


def create_lyrics_analysis_table(db_name: str) -> None:
    """
    Create the review_topic table in the SQLite database
    :param db_name: Name of table to create within the SQLite database
    :return: None
    """
    sql_statement = """
        CREATE TABLE IF NOT EXISTS lyrics_analysis (
                band TEXT NOT NULL, 
                song TEXT NOT NULL,
                album TEXT NOT NULL,
                year_recorded INT NOT NULL,
                lyrics TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                sentiment_grade FLOAT NOT NULL,
                PRIMARY KEY (band, song)
        );"""
    # Create a database connection
    try:
        with sqlite3.connect(db_name) as conn:
            cursor = conn.cursor()
            cursor.execute(sql_statement)
            conn.commit()
    except sqlite3.Error as e:
        print(e)


def song_in_table(conn: sqlite3.connect, band: str, song: str) -> bool:
    """
    :param conn: database connection
    :param band: Band name
    :param song: Song name
    return True if the band/song is already in the database, False otherwise
    """
    # Updating song and band strings to match what would be in table
    band_check = quote_updater(band)
    song_check = quote_updater(song)
    sql = f"SELECT COUNT(song) FROM lyrics_analysis WHERE band = '{band_check}' AND song = '{song_check}'"
    cur = conn.cursor()
    cur.execute(sql)
    num_rows = cur.fetchall()
    if num_rows[0][0] > 0:
        return True
    else:
        return False


def add_lyric_analysis(conn: sqlite3.connect, lyrics_data: tuple) -> None:
    """
    Add records to the lyrics_analysis table in the SQLite database
    :param conn: database connection
    :param lyrics_data: tuple containing row to insert into table
    :return:
    """
    sql = f"""INSERT INTO lyrics_analysis (band, song, album, year_recorded, lyrics, sentiment, sentiment_grade) 
              VALUES  {lyrics_data}"""
    cur = conn.cursor()
    cur.execute(sql)
    conn.commit()


def retrieve_song_sentiment(conn: sqlite3.connect, band: str, song: str) -> tuple[str, float]:
    """
    Retrieve records from the lyrics_analysis table in the SQLite database
    :param conn: database connection
    :param band: Band name for which to retrieve sentiment from the database
    :param song: Song title for which to retrieve sentiment from the database
    :return: Song's sentiment and sentiment grade between -1 and 1
    """
    # Updating song and band strings to match what would be in table
    band_check = quote_updater(band)
    song_check = quote_updater(song)
    sql = f"SELECT sentiment, sentiment_grade FROM lyrics_analysis WHERE band = '{band_check}' AND song = '{song_check}'"
    cur = conn.cursor()
    cur.execute(sql)
    sentiment, grade = cur.fetchall()[0]
    sentiment = sentiment.replace('""', '"').replace('"```', '```').replace('```"', '```').replace('\\n', '\n')
    return sentiment, grade


def quote_updater(text: str, errors: str = "strict") -> Union[str | None]:
    """
    This function inputs a text string and ensures it can be encoded as UTF-8, makes sure there are no NULL characters,
    replaces all "" with ", and removes the ' from contractions and wraps the entire string in double quotes, so it can
    be properly handled by our SQL database.

    :param text: Text to be analyzed and processed
    :param errors: Error type to be passed to codecs in processing Unicode text
    :return: Processed text or None if themes already None
    """
    if text:
        encodable = text.encode("utf-8", errors).decode("utf-8")

        nul_index = encodable.find("\x00")

        if nul_index >= 0:
            error = UnicodeEncodeError("utf-8", encodable, nul_index, nul_index + 1,
                                       "NULL not allowed")
            error_handler = codecs.lookup_error(errors)
            replacement, _ = error_handler(error)
            encodable = encodable.replace("\x00", replacement)

        encodable = "\"" + encodable.replace("\"", "\"\"") + "\""

        return encodable.replace("'", "")
    else:
        return None
