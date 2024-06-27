import json
import os
import feedparser
from textblob import TextBlob
import pandas as pd
from deep_translator import GoogleTranslator
import folium
from folium.plugins import HeatMap

def load_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from the configuration file: {file_path}")
        return None

def fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold, source_lang='da'):
    feed = feedparser.parse(url)
    translator = GoogleTranslator(source=source_lang, target='en')
    news_items = []
    for entry in feed.entries:
        original_title = entry.title
        try:
            translated_title = translator.translate(original_title)
            sentiment = TextBlob(translated_title).sentiment
            news_items.append([original_title, translated_title, sentiment.polarity, sentiment.subjectivity])
        except Exception as e:
            print(f"Error translating or analyzing entry: {original_title}, Error: {e}")
    df = pd.DataFrame(news_items, columns=['Original Title', 'Translated Title', 'Sentiment Polarity', 'Sentiment Subjectivity'])
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.width', 2000)
    df = df[(df['Sentiment Polarity'] > min_polarity_threshold) & (df['Sentiment Polarity'] < max_polarity_threshold)]
    return df

def analyze_all_news(config, min_polarity_threshold, max_polarity_threshold):
    all_news = []
    for country, regions in config.items():
        for region, urls in regions.items():
            for url in urls:
                df = fetch_rss_feed(url, min_polarity_threshold, max_polarity_threshold, source_lang='da')  # Change source_lang as needed
                df['Country'] = country
                df['Region'] = region
                all_news.append(df)
    if all_news:
        return pd.concat(all_news, ignore_index=True)
    else:
        return pd.DataFrame()

def generate_heatmap(df):
    if df.empty:
        print("No data to plot.")
        return
    
    # For simplicity, we use hardcoded coordinates for regions. In a real application, use a geocoding service.
    coordinates = {
        'Copenhagen': [55.6761, 12.5683],
        'Aarhus': [56.1629, 10.2039],
        'New York': [40.7128, -74.0060],
        'California': [36.7783, -119.4179]
    }
    
    heat_data = []
    for _, row in df.iterrows():
        region_coords = coordinates.get(row['Region'])
        if region_coords:
            heat_data.append([region_coords[0], region_coords[1], row['Sentiment Polarity']])
    
    # Create a map centered around the average location
    world_map = folium.Map(location=[20, 0], zoom_start=2)
    HeatMap(heat_data).add_to(world_map)
    
    # Save the map to an HTML file
    world_map.save('news_sentiment_heatmap.html')
    print("Heatmap saved to news_sentiment_heatmap.html")

def main():
    config_path = os.path.join(os.path.dirname(__file__), 'rss_feeds.json')
    config = load_config(config_path)
    if not config:
        return
    min_polarity_threshold = -0.5
    max_polarity_threshold = 0.5
    df = analyze_all_news(config, min_polarity_threshold, max_polarity_threshold)
    generate_heatmap(df)

if __name__ == "__main__":
    main()
