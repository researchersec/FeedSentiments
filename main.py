import json
import os
import feedparser
from textblob import TextBlob
import pandas as pd
from deep_translator import GoogleTranslator
import folium
from folium.plugins import HeatMap
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(file_path):
    try:
        logging.info(f"Loading configuration from {file_path}...")
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        logging.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from the configuration file: {file_path}")
        return None

def fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold, source_lang='da'):
    logging.info(f"Fetching RSS feed from {url}...")
    feed = feedparser.parse(url)
    translator = GoogleTranslator(source=source_lang, target='en')
    news_items = []
    for entry in feed.entries:
        original_title = entry.title
        try:
            logging.debug(f"Translating title: {original_title}")
            translated_title = translator.translate(original_title)
            sentiment = TextBlob(translated_title).sentiment
            logging.debug(f"Translated title: {translated_title}, Polarity: {sentiment.polarity}, Subjectivity: {sentiment.subjectivity}")
            news_items.append([original_title, translated_title, sentiment.polarity, sentiment.subjectivity])
        except Exception as e:
            logging.error(f"Error translating or analyzing entry: {original_title}, Error: {e}")
    df = pd.DataFrame(news_items, columns=['Original Title', 'Translated Title', 'Sentiment Polarity', 'Sentiment Subjectivity'])
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 2000)
    df = df[(df['Sentiment Polarity'] > min_polarity_threshold) & (df['Sentiment Polarity'] < max_polarity_threshold)]
    logging.info(f"Filtered {len(df)} articles based on sentiment polarity thresholds.")
    return df

def analyze_all_news(config, min_polarity_threshold, max_polarity_threshold):
    if not config:
        return pd.DataFrame()
    all_news = []
    for country, regions in config.items():
        logging.info(f"Analyzing news for country: {country}")
        for region, urls in regions.items():
            logging.info(f"Analyzing region: {region}")
            for url in urls:
                df = fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold, source_lang='da')  # Change source_lang as needed
                df['Country'] = country
                df['Region'] = region
                all_news.append(df)
    if all_news:
        logging.info(f"Combining data from {len(all_news)} regions.")
        return pd.concat(all_news, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_heatmap(df):
    if df.empty:
        logging.warning("No data to plot.")
        return
    
    logging.info("Generating heatmap...")
    coordinates = {
        'Copenhagen': [55.6761, 12.5683],
        'Aarhus': [56.1629, 10.2039],
        'New York': [40.7128, -74.0060],
        'California': [36.7783, -119.4179],
        # Add more regions and their coordinates as needed
    }
    
    heat_data = []
    for _, row in df.iterrows():
        region_coords = coordinates.get(row['Region'])
        if region_coords:
            heat_data.append([region_coords[0], region_coords[1], row['Sentiment Polarity']])
        else:
            logging.warning(f"No coordinates found for region: {row['Region']}")
    
    if not heat_data:
        logging.warning("No heat data to plot.")
        return
    
    world_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data).add_to(world_map)
    
    output_file = 'news_sentiment_heatmap.html'
    world_map.save(output_file)
    logging.info(f"Heatmap saved to {output_file}")

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'rss_feeds.json')
    config = load_config(config_path)
    if not config:
        return
    min_polarity_threshold = -0.5
    max_polarity_threshold = 0.5
    df = analyze_all_news(config, min_polarity_threshold, max_polarity_threshold)
    logging.info(f"Total articles analyzed: {len(df)}")
    generate_heatmap(df)

if __name__ == "__main__":
    main()
