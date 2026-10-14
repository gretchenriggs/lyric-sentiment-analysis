"""
This module contains constants and functions used to prompt the LLM to get responses related to the sentiment grade of
    a set of song lyrics.
"""

import json
import os
import time

from openai import OpenAI, OpenAIError

from lyrics_sentiment_analysis.sentiment_extraction import SentimentExtraction, SentimentGradeExtraction

LLM_MODEL = "gpt-4o-mini"
LYRIC_PROMPT = "Give me a one sentence summary of the sentiment of this song: "
ALBUM_PROMPT = ("Give me a one sentence summary of this list of song sentiments to give an overview of the "
                "sentiment for the entire album the songs appear on: ")
SENTIMENT_GRADE_PROMPT = """Analyze the following text thoroughly assign a sentiment grade between -1.0 and 1.0, where:
                            - -1.0 represents the most negative sentiment
                            - 0.0 represents a completely neutral sentiment
                            - 1.0 represents the most positive sentiment
                           Return the sentiment grade in the following JSON format:
                           {
                           "sentiment grade: [float-valued score between -1.0 and 1.0]
                           }:
                         """

CLIENT = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def delayed_completion(delay_in_seconds: float = 0.25, **kwargs) -> (
        CLIENT.chat.completions):
    """This function that adds a delay to a Completion API call
    :param delay_in_seconds: Number of seconds to delay between API calls
    return LLM model response
    """
    # Sleep for the delay
    time.sleep(delay_in_seconds)

    # Call the Completion API and return the result
    return CLIENT.beta.chat.completions.parse(**kwargs)


def get_sentiment(lyrics: str, temperature: float = 1.0, max_retries=3, backoff_base=2) -> str | None:
    """
    Find the sentiment of the lyrics
    :param lyrics: Song lyrics
    :param temperature: Temperature setting of LLM, higher temperature generates more creative and varied responses
    :param max_retries: Number of times to retry the LLM prompt if it fails
    :param backoff_base: Number to use as base in computing of wait time between retries (base to the retries power)
    :return: String containing sentiment of song lyrics
    """
    lyric_prompt = LYRIC_PROMPT + lyrics
    retries = 0
    while retries < max_retries:
        try:
            response = delayed_completion(model=LLM_MODEL,
                                          messages=[{"role": "user",
                                                     "content": lyric_prompt
                                                     }],
                                          max_tokens=256,
                                          temperature=temperature,
                                          response_format=SentimentExtraction
                                          )
            content = response.choices[0].message.content
            content = json.loads(content).get("app_features")
            return content
        except OpenAIError as e:
            wait_time = backoff_base ** retries
            print(f"API call failed (attempt {retries + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    print("Failed to get a response after multiple retries.")
    return None


def get_album_sentiment(song_sentiments: str, temperature: float = 1.0, max_retries=3, backoff_base=2) -> str | None:
    """
    Find the general sentiment of the album based on the sentiment of each song on the album
    :param song_sentiments: Sentiments for each song on the album
    :param temperature: Temperature setting of LLM, higher temperature generates more creative and varied responses
    :param max_retries: Number of times to retry the LLM prompt if it fails
    :param backoff_base: Number to use as base in computing of wait time between retries (base to the retries power)
    :return: Sentiment for the entire album
    """
    album_prompt = ALBUM_PROMPT + song_sentiments
    retries = 0
    while retries < max_retries:
        try:
            response = delayed_completion(model=LLM_MODEL,
                                          messages=[{"role": "user",
                                                     "content": album_prompt
                                                     }],
                                          max_tokens=256,
                                          temperature=temperature,
                                          response_format=SentimentExtraction
                                          )
            content = response.choices[0].message.content
            content = json.loads(content).get('app_features')
            return content
        except OpenAIError as e:
            wait_time = backoff_base ** retries
            print(f"API call failed (attempt {retries + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    print("Failed to get a response after multiple retries.")
    return None


def grade_sentiment(song_sentiment: str, temperature=0.5, max_retries=3, backoff_base=2) -> float | None:
    """
    Use LLM to grade sentiment on a scale from -1 to 1, with 0 being a neutral sentiment.

    :param song_sentiment: The sentiment of a song's lyrics, derived using LLM
    :param temperature: Temperature setting of LLM, higher temperature generates more creative and varied responses
    :param max_retries: Number of times to retry the LLM prompt if it fails
    :param backoff_base: Number to use as base in computing of wait time between retries (base to the retries power)
    :return: The grade of the song sentiment on a scale from -1 to 1
    """
    grade_prompt = SENTIMENT_GRADE_PROMPT + song_sentiment
    retries = 0
    while retries < max_retries:
        try:
            response = delayed_completion(model=LLM_MODEL,
                                          messages=[{"role": "user",
                                                     "content": grade_prompt
                                                     }],
                                          max_tokens=256,
                                          temperature=temperature,
                                          response_format=SentimentGradeExtraction
                                          )
            content = response.choices[0].message.content
            grade = json.loads(content).get('sentiment_grade')
            return float(grade)
        except OpenAIError as e:
            wait_time = backoff_base ** retries
            print(f"API call failed (attempt {retries + 1}/{max_retries}): {e}")
            print(f"Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
            retries += 1
    print("Failed to get a response after multiple retries.")
    return None
