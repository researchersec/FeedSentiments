import json
import os
import feedparser
from textblob import TextBlob
import pandas as pd
from deep_translator import GoogleTranslator

def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file: {file_path}")
        return None

def fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold):
    feed = feedparser.parse(url)
    translator = GoogleTranslator(source='da', target='en')
    news_items = []
    for entry in feed.entries:
        translated_title = translator.translate(entry.title)
        sentiment = TextBlob(translated_title).sentiment
        news_items.append([entry.title, translated_title, sentiment.polarity, sentiment.subjectivity])
    df = pd.DataFrame(news_items, columns=['Original Title', 'Translated Title', 'Sentiment Polarity', 'Sentiment Subjectivity'])
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 2000)
    df = df[(df['Sentiment Polarity'] > min_polarity_threshold) & (df['Sentiment Polarity'] < max_polarity_threshold)]
    return df

def analyze_country_news(config, country, min_polarity_threshold, max_polarity_threshold):
    if not config:
        return pd.DataFrame()
    country_feeds = config.get(country, {})
    all_news = []
    for region, urls in country_feeds.items():
        for url in urls:
            df = fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold)
            df['Region'] = region
            all_news.append(df)
    if all_news:
        return pd.concat(all_news, ignore_index=True)
    else:
        return pd.DataFrame()

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'rss_feeds.json')
    config = load_config(config_path)
    if not config:
        return
    country = 'Denmark'  # Example country
    min_polarity_threshold = -0.5
    max_polarity_threshold = 0.5
    df = analyze_country_news(config, country, min_polarity_threshold, max_polarity_threshold)
    print(df)

if __name__ == "__main__":
    main()
