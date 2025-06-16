"""
This is a developer module to create a visualizations cache seperatly from the streamlit app to handle 
computational restrictions on streamlit.
"""

import os
import pandas as pd
import pickle
import zipfile
from tqdm import tqdm
from app import str_to_list, list_to_str, remove_unbalanced_subtopics, radar, histogram, trend, heatmap

# Constants
FILE_PATH = "Data/classified_comment_stance_with_features.csv"
VIS_ZIP_PATH = "Viz/visualizations.zip"
VIS_PKL_NAME = "visualizations.pkl"


def load_and_process_data(csv_filepath, sample=None):
    df = pd.read_csv(csv_filepath, index_col=None, on_bad_lines='skip')
    if sample:
        df = df.sample(n=sample, random_state=42)

    columns_to_keep = [
        "comment_id", "created_time", "score", "predicted_label", "toxic", "severe_toxic",
        "obscene", "threat", "insult", "identity_hate", "sentiment_score", "factual_score",
        "belief_score", "emotionality_score", "super_topics"
    ]
    df = df[columns_to_keep]
    df.rename(columns={
        "score": "Score",
        "predicted_label": "Affiliation",
        "toxic": "Toxicity Score",
        "severe_toxic": "Severe Toxicity Score",
        "obscene": "Obscenity Score",
        "threat": "Threat Score",
        "insult": "Insult Score",
        "identity_hate": "Identity Hate Score",
        "sentiment_score": "Polarity Sentiment Score",
        "factual_score": "Factual Speech Similarity",
        "belief_score": "Belief Speech Similarity",
        "emotionality_score": "Emotionality Score",
        "super_topics": "Topics"
    }, inplace=True)

    df['Topics'] = df['Topics'].apply(str_to_list)
    df = df[df['Topics'].apply(lambda x: len(x) > 0)].reset_index(drop=True)
    df = remove_unbalanced_subtopics(df)
    df['Topics'] = df['Topics'].apply(list_to_str)
    df = df[df['Affiliation'].isin({'Pro-Israel', 'Pro-Palestine'})]

    numeric_cols = [
        "Score", "Toxicity Score", "Severe Toxicity Score", "Obscenity Score",
        "Threat Score", "Insult Score", "Identity Hate Score", "Polarity Sentiment Score",
        "Factual Speech Similarity", "Belief Speech Similarity", "Emotionality Score"
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(how='any').reset_index(drop=True)

    if df.empty:
        raise ValueError("Processed DataFrame is empty after cleaning.")

    return df


def precompute_visualizations(df):
    subtopics = ['Overall'] + sorted(set(
        t for topics in df['Topics'].apply(str_to_list) for t in topics
    ))
    features = [
        "Toxicity Score", "Severe Toxicity Score", "Obscenity Score", "Threat Score",
        "Insult Score", "Identity Hate Score", "Polarity Sentiment Score",
        "Emotionality Score", "Factual Speech Similarity", "Belief Speech Similarity"
    ]

    total = len(subtopics) * len(features) + len(features)  # all radars + all subtopic-feature combos
    pbar = tqdm(total=total, desc="Generating Visualizations")

    visualizations = {}
    radars = {}
    for feature in features:
        radars[feature] = radar(df, feature)
        pbar.update(1)

    for subtopic in subtopics:
        visualizations[subtopic] = {}
        heatmap_fig = heatmap(df, subtopic)
        for feature in features:
            visualizations[subtopic][feature] = {
                'heatmap': heatmap_fig,
                'histogram': histogram(df, subtopic, feature),
                'trend': trend(df, subtopic, feature),
                'radar': radars[feature]
            }
            pbar.update(1)

    pbar.close()
    return visualizations


def save_visualizations_to_zip(visualizations, zip_path=VIS_ZIP_PATH):
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        with zipf.open(VIS_PKL_NAME, 'w') as f:
            pickle.dump(visualizations, f)


def run():
    print("📥 Loading and processing dataset...")
    df = load_and_process_data(FILE_PATH)
    print("✅ Data processed.")

    print("📊 Computing visualizations...")
    visualizations = precompute_visualizations(df)
    print("✅ Visualizations computed.")

    print(f"💾 Saving to {VIS_ZIP_PATH}...")
    save_visualizations_to_zip(visualizations)
    print("✅ All done!")


if __name__ == "__main__":
    run()