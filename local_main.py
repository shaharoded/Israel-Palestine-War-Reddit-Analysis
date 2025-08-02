"""
This module is intended to create the viz locally so they can later be loaded to drive properly
"""
import os
import pickle
import zipfile
from tqdm import tqdm
from app import radar, histogram, trend, heatmap, pie_chart, load_and_process_data, str_to_list

import plotly.io as pio
pio.renderers.default = 'browser'

def precompute_visualizations(df):
    '''
    Pre-compute all visualizations with CLI progress tracking using tqdm.
    Returns a nested dictionary of visualizations.
    '''
    subtopics = ['Overall'] + sorted(set(
        t for topics in df['Topics'].apply(str_to_list) for t in topics
    ))
    
    features = [
        "Toxicity Score", "Severe Toxicity Score", "Obscenity Score", "Threat Score",
        "Insult Score", "Identity Hate Score", "Polarity Sentiment Score",
        "Emotionality Score", "Factual Speech Similarity", "Belief Speech Similarity"
    ]

    total_steps = len(features) + len(subtopics) * len(features)
    progress = tqdm(total=total_steps, desc="Precomputing Visualizations")

    visualizations = {}
    radars = {}
    
    # Compute pie-chart
    pie_fig = pie_chart(data_dict = {
        'Pro-Israel': 440587,
        'Pro-Palestine': 407135,
        'Unclassified': 1850208})
    
    # Precompute radar plots once per feature
    for feature in features:
        radars[feature] = radar(df, feature)
        progress.update(1)
    
    # Compute plots based on sub-topic
    for subtopic in subtopics:
        visualizations[subtopic] = {}
        heatmap_fig = heatmap(df, subtopic)
        # Compute based on feature
        for feature in features:
            radar_fig = radars[feature]
            histogram_fig = histogram(df, subtopic, feature)
            trend_fig = trend(df, subtopic, feature)
            visualizations[subtopic][feature] = {
                'heatmap': heatmap_fig,
                'histogram': histogram_fig,
                'trend': trend_fig,
                'radar': radar_fig
            }
            progress.update(1)

    progress.close()
    
    # Add score averages to the visualization dictionary
    pro_israel_score = df[df['Affiliation'] == 'Pro-Israel']['Score'].mean()
    pro_palestine_score = df[df['Affiliation'] == 'Pro-Palestine']['Score'].mean()
    visualizations["_meta"] = {
        "pro_israel_score": pro_israel_score,
        "pro_palestine_score": pro_palestine_score,
        "pie": pie_fig
    }

    return visualizations


def save_visualizations_locally():
    FILE_PATH = "Data/classified_comment_stance_with_features.csv"
    VIS_ZIP_PATH = "Viz/visualizations.zip"
    pickle_filename = "visualizations.pkl"

    # Ensure output directory exists
    os.makedirs(os.path.dirname(VIS_ZIP_PATH), exist_ok=True)

    print("🔄 Loading and processing dataset...")
    df = load_and_process_data(FILE_PATH, sample=5000)

    print("📊 Precomputing visualizations...")
    visualizations = precompute_visualizations(df)

    print("💾 Saving visualizations to pickle...")
    with open(pickle_filename, "wb") as f:
        pickle.dump(visualizations, f)

    print("📦 Zipping visualizations...")
    with zipfile.ZipFile(VIS_ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(pickle_filename, arcname=os.path.basename(pickle_filename))

    os.remove(pickle_filename)
    print(f"✅ Visualization ZIP saved to: {VIS_ZIP_PATH}")



if __name__ == "__main__":
    save_visualizations_locally()