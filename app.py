import os
import re
import gdown
import pandas as pd
import numpy as np
import ast
from datetime import datetime
import zipfile
import pickle
import warnings
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ignore PerformanceWarning
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Google Drive viz zip file ID
VIS_ZIP_PATH = "Viz/visualizations.zip"
VIS_ZIP_GDRIVE_ID = "1fxEWLwRDQztM-hWYPG293RZvDNdvoE3w"  # replace with your actual file ID
VIS_ZIP_DOWNLOAD_URL = f"https://drive.google.com/uc?id={VIS_ZIP_GDRIVE_ID}"

# Google Drive data file ID
FILE_PATH = "Data/classified_comment_stance_with_features.csv"
FILE_ID = "1J7rrdBLdve0JM0yGWwygrs-i1dB6cC4q"
DOWNLOAD_URL = f"https://drive.google.com/uc?id={FILE_ID}"


def radar(data, column):
    def wrap_topic(label: str, max_word=9, split_at=7):
        """
        • Replace spaces, hyphens, slashes with line-breaks (<br>)
        • If any remaining word segment still runs longer than `max_word`,
        split after `split_at` chars, add a hyphen, drop to next line.
        """
        # first pass: break on delimiters
        parts = re.sub(r"[ \-/]", "<br>", label).split("<br>")

        wrapped = []
        for part in parts:
            while len(part) > max_word:
                wrapped.append(part[:split_at] + "-")
                part = part[split_at:]
            wrapped.append(part)
        return "<br>".join(wrapped)   
    # Preprocess the data
    data = data.copy()
    data = data[data['Topics'].apply(bool)]

    # Explode the data to have one row per subtopic
    data['Topics'] = data['Topics'].apply(str_to_list)
    exploded_data = data.explode('Topics')
    exploded_data = exploded_data.dropna(subset=['Topics'])

    # Group by Affiliation and Subtopics to calculate the average sentiment
    grouped_data = exploded_data.groupby(['Affiliation', 'Topics']).agg({column: 'mean'}).reset_index()

    # Prepare data for radar plot by group
    pro_israel_data = grouped_data[grouped_data['Affiliation'] == 'Pro-Israel']
    pro_palestine_data = grouped_data[grouped_data['Affiliation'] == 'Pro-Palestine']

    # Ensure Subtopics are aligned between the two groups for consistent radar plot structure
    all_subtopics = set(pro_israel_data['Topics']).union(set(pro_palestine_data['Topics']))
    for subtopic in all_subtopics:
        if subtopic not in pro_israel_data['Topics'].values:
            pro_israel_data = pd.concat([pro_israel_data, pd.DataFrame([{
                'Affiliation': 'Pro-Israel',
                'Topics': subtopic,
                column: 0
            }])], ignore_index=True)
        if subtopic not in pro_palestine_data['Topics'].values:
            pro_palestine_data = pd.concat([pro_palestine_data, pd.DataFrame([{
                'Affiliation': 'Pro-Palestine',
                'Topics': subtopic,
                column: 0
            }])], ignore_index=True)

    # Sort by Subtopics to ensure consistency
    pro_israel_data = pro_israel_data.sort_values(by='Topics')
    pro_palestine_data = pro_palestine_data.sort_values(by='Topics')

    subtopics_israel_raw      = pro_israel_data['Topics'].tolist()
    subtopics_palestine_raw   = pro_palestine_data['Topics'].tolist()

    # wrap long labels → line-break friendly versions
    subtopics_israel     = [wrap_topic(t) for t in subtopics_israel_raw]
    subtopics_palestine  = [wrap_topic(t) for t in subtopics_palestine_raw]

    values_israel        = pro_israel_data[column].tolist()
    values_palestine     = pro_palestine_data[column].tolist()

    # Create DataFrames for Plotly
    df_israel = pd.DataFrame(dict(
        r=values_israel + [values_israel[0]],  # Close the loop
        theta=subtopics_israel + [subtopics_israel[0]]  # Close the loop
    ))

    df_palestine = pd.DataFrame(dict(
        r=values_palestine + [values_palestine[0]],  # Close the loop
        theta=subtopics_palestine + [subtopics_palestine[0]]  # Close the loop
    ))

    # Create the radar chart for pro-Israel
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=df_israel['r'],
        theta=df_israel['theta'],
        fill='toself',
        name='Pro-Israel',
        line=dict(color='rgba(0, 0, 255, 0.6)'),
        hovertemplate=f'Pro-Israel<br>Avg {column}: %{{r}}<br>Topic: %{{theta}}<extra></extra>'
    ))

    # Add the radar chart for pro-Palestine
    fig.add_trace(go.Scatterpolar(
        r=df_palestine['r'],
        theta=df_palestine['theta'],
        fill='toself',
        name='Pro-Palestine',
        line=dict(color='rgba(0, 128, 0, 0.6)'),
        hovertemplate=f'Pro-Palestine<br>Avg {column}: %{{r}}<br>Topic: %{{theta}}<extra></extra>'
    ))

    # Update layout for title and axis labels
    fig.update_layout(
        showlegend=False,
        legend=dict(
            font=dict(size=12, color='#454A4A')  # Update legend font color
        ),
        polar=dict(
            radialaxis=dict(visible=True, range=[min(values_israel + values_palestine), max(values_israel + values_palestine)], tickfont=dict(size=10, color='#454A4A')),
            angularaxis=dict(
                tickfont=dict(size=10, color='#454A4A'),
                categoryarray=subtopics_israel + [subtopics_israel[0]],  # Set custom category order
                categoryorder='array'
            )
        ),
        hoverlabel=dict(font_size=14, font_color='#454A4A'),  # Increased font size and updated color for hover text
        width=400,  # Set the figure width
        height=400,  # Set the figure height
        margin=dict(t=10, b=10, l=50, r=50)  # Adjusted margins
    )

    return fig


def histogram(data, selected_subtopic, column):
    data = data.copy()
    data = data[data['Topics'].apply(bool)]
    
    # Create subset of the data based on subtopic
    if selected_subtopic != "Overall":
        data['Topics'] = data['Topics'].apply(str_to_list)
        data = data.explode('Topics')
        data = data[data['Topics'] == selected_subtopic]
    
    # Guard from edge cases
    if data.empty or data[column].dropna().empty:
        return go.Figure()
    
    # Define bins for scores. 10 bins in the viz
    # get the boundries per score 
    _min = np.floor(data[column].min())
    _max = np.ceil(data[column].max())
    # Check if _min is equal to _max to avoid zero division
    if _min == _max:
        raise ValueError(f'''Min and Max Values in column {column} are the same = {_max}. Check data. 
                         Data length = {len(data)}, 
                         columns = {data.columns},
                         subtopic = {selected_subtopic}
                         ''')
    else:
        bin_size = (_max - _min) / 10
        _max = _max + bin_size
        bins = np.arange(_min, _max, bin_size)  # Bins from int(min) to int(max) with step bin_size

    # Separate data for Pro-Israel and Pro-Palestine
    pro_israel_df = data[data['Affiliation'] == 'Pro-Israel']
    pro_palestine_df = data[data['Affiliation'] == 'Pro-Palestine']

    # Create a figure
    fig = make_subplots(rows=1, cols=1)

    # Get data for the selected subtopic
    pro_israel_data = pro_israel_df[column]
    pro_palestine_data = pro_palestine_df[column]

    # Calculate the total number of records for each group and topic
    total_pro_israel = len(pro_israel_data)
    total_pro_palestine = len(pro_palestine_data)

    # Create histograms
    counts_pro_israel, _ = np.histogram(pro_israel_data, bins=bins)
    counts_pro_palestine, _ = np.histogram(pro_palestine_data, bins=bins)

    # Normalize counts to reflect percentages within each group and topic
    perc_pro_israel = (counts_pro_israel / total_pro_israel * 100) if total_pro_israel > 0 else np.zeros(len(counts_pro_israel))
    perc_pro_palestine = (counts_pro_palestine / total_pro_palestine * 100) if total_pro_palestine > 0 else np.zeros(len(counts_pro_palestine))

    # Define bin centers for plotting
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Create hover text
    hover_text_pro_israel = [f"Scores in Bin: {bins[i]:.1f} to {bins[i+1]:.1f},\n{perc_pro_israel[i]:.1f}% of the group's comments" for i in range(len(perc_pro_israel))]
    hover_text_pro_palestine = [f"Scores in Bin: {bins[i]:.1f} to {bins[i+1]:.1f},\n{perc_pro_palestine[i]:.1f}% of the group's comments" for i in range(len(perc_pro_palestine))]

    # Create traces
    trace_israel = go.Bar(
        x=bin_centers,
        y=perc_pro_israel,
        name='Pro-Israel',
        marker_color='rgba(0, 0, 255, 0.6)',
        opacity=0.7,
        width=bin_size/2,
        hovertext=hover_text_pro_israel,
        hoverinfo='text'
    )
    trace_palestine = go.Bar(
        x=bin_centers,
        y=perc_pro_palestine,
        name='Pro-Palestine',
        marker_color='rgba(0, 128, 0, 0.6)',
        opacity=0.7,
        width=bin_size/2,
        hovertext=hover_text_pro_palestine,
        hoverinfo='text'
    )

    fig.add_traces([trace_israel, trace_palestine])

    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",      # <-- side-by-side items
            yanchor="bottom",
            y=1.08,               # a bit above the top axis
            xanchor="center",
            x=0.5,
            font=dict(size=16, color="#454A4A")
        ),

        xaxis=dict(
            title="",            
            tickfont=dict(color="#454A4A", size=12) 
        ),

        # enlarge y-axis tick labels
        yaxis=dict(
            title=dict(text="Percentage of Comments", font=dict(color="#454A4A")),
            tickfont=dict(color="#454A4A", size=12),
            tickvals=[0,10,20,30,40,50,60,70,80,90,100],
            ticktext=["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"]
        ),

        barmode="group",
        margin=dict(t=10, b=0, l=50, r=50),
        hoverlabel=dict(font_size=14, font_color="#454A4A"),
        height=300
    )
    return fig


def trend(data, selected_subtopic, column, agg='mean'):
    data = data.copy()
    data = data[data['Topics'].apply(bool)]
    
    # Create subset of the data based on subtopic
    if selected_subtopic != "Overall":
        data['Topics'] = data['Topics'].apply(str_to_list)
        data = data.explode('Topics')
        data = data[data['Topics'] == selected_subtopic]
    
    # Guard from edge cases
    if data.empty:
        return go.Figure()
    if agg == 'mean' and data[column].dropna().empty:
        return go.Figure()
    
    data['created_time'] = pd.to_datetime(data['created_time'])
    data['month'] = data['created_time'].dt.to_period('M').dt.to_timestamp()

    if agg == 'count':
        grouped = data.groupby(['month', 'Affiliation']).size().reset_index(name=column)
    else:  # default to mean
        grouped = data.groupby(['month', 'Affiliation'])[column].mean().reset_index()

    fig = px.line(
        grouped,
        x='month',
        y=column,
        color='Affiliation',
        color_discrete_map={
            'Pro-Israel': '#003f5c',
            'Pro-Palestine': '#2f9e44'
        },
        markers=True,
        line_shape='spline'
    )

    fig.update_layout(
        title_text="",
        xaxis_title='Month',
        yaxis_title=f'Average {column}' if agg == 'mean' else 'Number of Comments',
        title_x=0.5,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            tickformat="%b\n%Y",  # e.g., Jan\n2025
            dtick="M1",           # force monthly ticks
            tickfont=dict(size=10, color="#454A4A")
        ),
        legend_title_text="Affiliation",                # ← keep this
        legend=dict(                                   # ← ADD this block
        orientation="h",   # horizontal
        yanchor="bottom",
        y=1.02,            # a bit above the top axis
        xanchor="center",
        x=0.5
        )
    )

    # Timeline events
    events = [
        ("Oct 7th\nAttack", datetime(2023, 10, 7)),
        ("First\nCeasefire", datetime(2023, 11, 24)),
        ("Rafah\nOperation", datetime(2024, 5, 7)),
        ("Beeper\nOperation", datetime(2024, 9, 17)),
        ("Sinwar\nAssassination", datetime(2024, 10, 17)),
        ("Merkavot\nGideon", datetime(2025, 5, 16)),
    ]

    for label, event_date in events:
        # Add vertical line
        fig.add_vline(
            x=event_date,
            line_width=1,
            line_dash="dot",
            line_color="gray"
        )

        # Add an invisible scatter point for hover text
        fig.add_trace(go.Scatter(
            x=[event_date],
            y=[grouped[column].max()],  # Place at top of visible y range
            mode='markers',
            marker=dict(color='red', size=6, symbol='line-ns-open'),
            name=label,
            hovertemplate=f"{label}<br>Date: {event_date.strftime('%Y-%m-%d')}<extra></extra>",
            showlegend=False
        ))

    return fig


def heatmap(df, subtopic):
    data = df.copy()

    if subtopic != "Overall":
        data['Topics'] = data['Topics'].apply(str_to_list)
        data = data.explode('Topics')
        data = data[data['Topics'] == subtopic]

    # Function to create hexbin traces
    def create_hexbin_trace(df, affiliation, color):
        subset = df[df['Affiliation'] == affiliation]

        x = subset['Factual Speech Similarity']
        y = subset['Belief Speech Similarity']

        hist, xedges, yedges = np.histogram2d(x, y, bins=[20, 20], range=[[0, 1], [0, 1]])
        x_centres = (xedges[:-1] + xedges[1:]) / 2   # length 20
        y_centres = (yedges[:-1] + yedges[1:]) / 2   # length 20
        hist = hist.T  # Transpose for correct plot orientation
        hist_sum = hist.sum()
        hist_percentile = (hist / hist.max()) * 100 if hist.max() > 0 else np.zeros_like(hist)
        hist_percentage = (hist / hist_sum) * 100 if hist_sum > 0 else np.zeros_like(hist)
        hist_percentage = hist_percentage.tolist()

        trace = go.Heatmap(
            x=x_centres,
            y=y_centres,
            z=hist_percentile,
            colorscale=color,
            showscale=False,
            name=f'{affiliation}',
            customdata=hist_percentage,  # Add custom data for the hover info
            hovertemplate=(
                'Factual Speech Similarity: %{x}<br>'
                'Belief Speech Similarity: %{y}<br>'
                'Density Measure (Percentile): %{z:.2f}%<br>'
                'Percent of Group: %{customdata:.2f}%<extra></extra>'
            ),
        )
        return trace

    # Create the figure with 2 subplots
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Pro-Israel', 'Pro-Palestine'], horizontal_spacing=0.15)

    # Colors for the hexbin plots
    colors = ['Blues', 'Greens']

    # Add traces for each affiliation
    for i, (affiliation, color) in enumerate(zip(['Pro-Israel', 'Pro-Palestine'], colors)):
        trace = create_hexbin_trace(data, affiliation, color)
        fig.add_trace(trace, row=1, col=i + 1)

    # Update layout
    fig.update_layout(
        xaxis=dict(
            range=[0, 1],
            title=dict(text='Factual Speech Similarity', font=dict(color='#454A4A')),
            tickfont=dict(color='#454A4A'),  # Change x-axis tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        yaxis=dict(
            range=[0, 1],
            title=dict(text='Belief Speech Similarity', font=dict(color='#454A4A')),
            tickfont=dict(color='#454A4A'),  # Change y-axis tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        xaxis2=dict(
            range=[0, 1],
            title=dict(text='Factual Speech Similarity', font=dict(color='#454A4A')),
            tickfont=dict(color='#454A4A'),  # Change x-axis2 tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        yaxis2=dict(
            range=[0, 1],
            title=dict(text='Belief Speech Similarity', font=dict(color='#454A4A')),
            tickfont=dict(color='#454A4A'),  # Change y-axis2 tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=80),
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to be transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background to be transparent
        font=dict(color='#454A4A'),
        hoverlabel=dict(font_size=14, font_color='#454A4A')  # Set hover label font size and color
    )

    # Ensure the grid is visible and fits properly
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#454A4A', zeroline=False, dtick=0.2)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#454A4A', zeroline=False, dtick=0.2)

    fig.update_layout(
    xaxis = dict(title=dict(text='Factual Speech Similarity',
                            font=dict(color='#454A4A', size=16))),
    yaxis = dict(title=dict(text='Belief Speech Similarity',
                            font=dict(color='#454A4A', size=16))),
    xaxis2 = dict(title=dict(text='Factual Speech Similarity',
                             font=dict(color='#454A4A', size=16))),
    yaxis2 = dict(title=dict(text='Belief Speech Similarity',
                             font=dict(color='#454A4A', size=16))),
    )

    fig.update_annotations(font=dict(size=16))

    return fig


def pie_chart(data_dict):
    # Convert the dictionary to lists for labels and sizes
    labels = list(data_dict.keys())
    sizes = list(data_dict.values())

    # Define colors
    colors = ['rgba(0, 0, 139, 0.3)', 'rgba(0, 128, 0, 0.3)', 'rgba(169, 169, 169, 0.3)']  # Dark grey for 'Unclassified'

    # Create hover text
    hover_text = [
        f'{size / sum(sizes) * 100:.2f}% of the Comments'
        for size in sizes
    ]

    # Sort the slices to start with the smaller one
    sorted_indices = np.argsort(sizes)[::-1]
    sizes = np.array(sizes)[sorted_indices]
    labels = np.array(labels)[sorted_indices]
    colors = np.array(colors)[sorted_indices]
    hover_text = np.array(hover_text)[sorted_indices]

    # Plotting the donut chart
    fig = go.Figure(data=[go.Pie(labels=labels, values=sizes, hole=0.70, hoverinfo='label+text', text=hover_text, textinfo='none', marker=dict(colors=colors))])

    fig.update_layout(
        annotations=[
            dict(
                text="<b>Classification Distribution</b>",
                x=0.5,
                y=1.25,
                font=dict(size=14, color="#454A4A"),
                showarrow=False,
                xanchor='center'
            )
        ],
        showlegend=False,  # Remove legend
        hoverlabel=dict(font_size=14, font_color='#454A4A'),  # Set hover label font size and color
        height=150,  # Adjust height
        margin=dict(l=10, r=10, t=30, b=10)  # Adjust margins
    )
    return fig


# Function to parse the Topics string
def list_to_str(topics):
    """Convert a list to a string for storage."""
    return str(topics if isinstance(topics, list) else [])


def str_to_list(topics_str):
    """Convert a stringified list (e.g., "[...]" or "[]") to a Python list."""
    if isinstance(topics_str, list):
        return topics_str
    if isinstance(topics_str, str) and topics_str.strip().startswith('['):
        try:
            return ast.literal_eval(topics_str)
        except (ValueError, SyntaxError):
            return []
    return []


def remove_unbalanced_subtopics(df):
    """
    Remove subtopics that appear only in one affiliation group.
    Operates in-place on the 'Topics' column, assuming it's a list of topics.
    """
    df = df.copy()
    df = df[df['Topics'].apply(bool)]
    df['Topics'] = df['Topics'].apply(str_to_list)
    exploded = df.explode('Topics')

    # Count unique affiliations per topic
    affiliation_counts = exploded.groupby('Topics')['Affiliation'].nunique()

    # Keep only topics appearing in both groups
    valid_topics = affiliation_counts[affiliation_counts > 1].index

    # Filter the original DataFrame
    df['Topics'] = df['Topics'].apply(lambda topics: [t for t in topics if t in valid_topics])

    # Optionally, drop rows that now have no topics left
    df = df[df['Topics'].apply(bool)]

    return df


@st.cache_data
def load_and_process_data(csv_filepath, sample=None):
    '''
    Pre-process the data, and cache to save calculations.
    Includes validation on Topics column format.

    sample (int): For local tests on small subsets of data.
    '''
    try:
        # Read CSV
        df = pd.read_csv(csv_filepath, index_col=None, on_bad_lines='skip')
        if sample:
            df = df.sample(n=sample, random_state=42)
        
        df['created_time'] = pd.to_datetime(df['created_time'], errors='coerce')
        start_date = pd.Timestamp('2023-10-06')   # keep Oct 6 2023 and later
        end_date   = pd.Timestamp('2025-05-31')   # keep up to May 31 2025
        df = df[(df['created_time'] >= start_date) & (df['created_time'] <= end_date)]

        # Keep only relevant columns and rename
        columns_to_keep = ["comment_id", "created_time", "score", "predicted_label", "toxic", "severe_toxic",
                           "obscene", "threat", "insult", "identity_hate", "sentiment_score", "factual_score", "belief_score",
                           "emotionality_score", "selected_topics"]
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
            "selected_topics": "Topics"
        }, inplace=True)
        print("✅ Renamed columns:", df.columns.tolist())

        # Ensure Topics are proper lists
        df['Topics'] = df['Topics'].apply(str_to_list)

        # Validate Topics column: all rows must be lists
        if df['Topics'].isnull().any():
            raise ValueError("Some rows in 'Topics' are null after parsing.")
        if not df['Topics'].apply(lambda x: isinstance(x, list)).all():
            raise ValueError("Non-list values found in 'Topics'.")
        if df['Topics'].apply(len).eq(0).all():
            raise ValueError("All Topics lists are empty.")
        
        # Drop rows with empty topic lists
        df = df[df['Topics'].apply(lambda x: len(x) > 0)].reset_index(drop=True)

        # Remove subtopics that only appear for one group
        df = remove_unbalanced_subtopics(df)

        # Re-stringify Topics (for caching purposes)
        df['Topics'] = df['Topics'].apply(list_to_str)

        # Keep only valid affiliations
        df = df[df['Affiliation'].isin({'Pro-Israel', 'Pro-Palestine'})]

        # Convert numeric columns
        numeric_cols = [
            "Score", "Toxicity Score", "Severe Toxicity Score", "Obscenity Score",
            "Threat Score", "Insult Score", "Identity Hate Score", "Polarity Sentiment Score",
            "Factual Speech Similarity", "Belief Speech Similarity", "Emotionality Score"
        ]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        df = df.dropna(how='any').reset_index(drop=True)

        # Final sanity check
        if df.empty:
            raise ValueError("Processed DataFrame is empty after cleaning.")
        
        return df

    except Exception as e:
        raise Exception(f"❌ Error loading dataset: {e}")


@st.cache_resource
def precompute_visualizations(df):
    '''
    Pre-compute all visualizations to avoid heavy calculation for every filter change.
    Displays a progress bar in the Streamlit UI.
    '''
    placeholder = st.empty()
    placeholder.subheader("")  # temporarily occupies space

    subtopics = ['Overall'] + sorted(set(
    t for topics in df['Topics'].apply(str_to_list) for t in topics
    ))
    features = [
        "Toxicity Score", "Severe Toxicity Score", "Obscenity Score", "Threat Score",
        "Insult Score", "Identity Hate Score", "Polarity Sentiment Score",
        "Emotionality Score", "Factual Speech Similarity", "Belief Speech Similarity"
    ]
    
    total_steps = len(features) + len(subtopics) * len(features)
    progress_bar = st.progress(0)
    step = 0

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
        step += 1
        progress_bar.progress(step / total_steps)

    # Compute plots based on sub-topic
    for subtopic in subtopics:
        visualizations[subtopic] = {}
        heatmap_fig = heatmap(df, subtopic)
        comment_trend_fig = trend(df, subtopic, 'comment_id', agg='count')
        for feature in features:
            radar_fig = radars[feature]
            histogram_fig = histogram(df, subtopic, feature)
            trend_fig = trend(df, subtopic, feature)
            visualizations[subtopic][feature] = {
                'heatmap': heatmap_fig,
                'histogram': histogram_fig,
                'trend': trend_fig,
                'radar': radar_fig,
                'comment_trend': comment_trend_fig
            }
            step += 1
            progress_bar.progress(step / total_steps)

    progress_bar.empty()  # remove the progress bar and place holder
    placeholder.empty()

    # Add score averages to the visualization dictionary
    pro_israel_score = df[df['Affiliation'] == 'Pro-Israel']['Score'].mean()
    pro_palestine_score = df[df['Affiliation'] == 'Pro-Palestine']['Score'].mean()
    visualizations["_meta"] = {
        "pro_israel_score": pro_israel_score,
        "pro_palestine_score": pro_palestine_score,
        "pie": pie_fig
    }

    return visualizations


def main():
    text_color = '#8E6C1E'
    small_text_color = '#454A4A'
    select_box_css = f"""
    <style>
        /* Style the select box container */
        .stSelectbox [data-baseweb="select"] {{
            border: 4px solid; /* Adjust border width */
            border-image: linear-gradient(to right, lightblue, lightgreen) 1; /* Lighter gradient border */
            border-radius: 10px; /* Rounded corners */
            background-color: #F0F8FF; /* Light background color */
        }}
        /* Style the select box text */
        .stSelectbox [data-baseweb="select"] .css-1hwfws3 {{
            color: #008000; /* Green text */
            font-size: 1.5rem; /* Larger font size */
            text-align: center; /* Centralized text */
        }}
        /* Adjust the font size of the selected option */
        .stSelectbox [data-baseweb="select"] .css-1wa3eu0-placeholder, 
        .stSelectbox [data-baseweb="select"] .css-1uccc91-singleValue {{
            font-size: 1.5rem; /* Larger font size for selected option */
            color: {text_color}; /* Text color */
            text-align: center; /* Centralized text */
        }}
        /* General text color */
        .main-text {{
            color: {text_color};
        }}
    </style>
    """

    information_hover = {
        'Polarity Sentiment Score': (
            'Polarity Sentiment Score (from RoBERTa Sentiment): This score ranges from -1 (very negative) to 1 (very positive) '
            'and captures the overall sentiment polarity of the comment.'
        ),
        'Toxicity Score': (
            'Toxicity Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment contains toxic or harmful language.'
        ),
        'Severe Toxicity Score': (
            'Severe Toxicity Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment is highly toxic or severely harmful.'
        ),
        'Obscenity Score': (
            'Obscenity Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment contains obscene or profane language.'
        ),
        'Threat Score': (
            'Threat Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment includes threats or intimidation.'
        ),
        'Insult Score': (
            'Insult Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment contains insults or derogatory language.'
        ),
        'Identity Hate Score': (
            'Identity Hate Score (from BERT Toxicity classifier): Probability (0 to 1) that the comment contains hate speech toward identity groups.'
        ),
        'Belief Speech Similarity': (
            'Belief Speech Similarity (from Word2Vec-based cosine similarity): Measures how closely the comment matches belief-based language patterns. Ranges from 0 (not belief-driven) to 1 (strong belief expression).'
        ),
        'Factual Speech Similarity': (
            'Factual Speech Similarity (from Word2Vec-based cosine similarity): Measures how closely the comment matches factual, objective language. Ranges from 0 (non-factual) to 1 (highly factual).'
        ),
        'Emotionality Score': (
            'Emotionality Score (from Word2Vec projection): Measures the emotional tone of the comment, with higher scores indicating more emotional content. Ranges from 0 to 1.'
        )
    }

    # Viz acquisition block 
    # Try loading precomputed visualizations FIRST
    visualizations = None

    if os.path.exists(VIS_ZIP_PATH):
        st.success("📦 Using locally cached visualizations.")
        try:
            with zipfile.ZipFile(VIS_ZIP_PATH, 'r') as zipf:
                with zipf.open("visualizations.pkl") as f:
                    visualizations = pickle.load(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to load local visualizations: {e}. Will attempt fallback...")

    # If no valid local visualizations, try downloading from Google Drive
    if visualizations is None:
        st.warning("📥 Attempting to download visualizations from Google Drive...")
        try:
            os.makedirs(os.path.dirname(VIS_ZIP_PATH), exist_ok=True)
            gdown.download(VIS_ZIP_DOWNLOAD_URL, VIS_ZIP_PATH, quiet=False)
            with zipfile.ZipFile(VIS_ZIP_PATH, 'r') as zipf:
                with zipf.open("visualizations.pkl") as f:
                    visualizations = pickle.load(f)
            st.success("✅ Downloaded and loaded visualizations from Google Drive.")
        except Exception as e:
            st.error(f"❌ Download failed: {e}. Falling back to local computation.")

    # If both local and remote visualizations failed, load + compute
    if visualizations is None:
        # Load dataset (only now)
        if not os.path.exists(FILE_PATH):
            os.makedirs(os.path.dirname(FILE_PATH), exist_ok=True)
            with st.status("📥 Downloading data from Google Drive... Please wait (~3 min).", expanded=True) as status:
                try:
                    gdown.download(DOWNLOAD_URL, FILE_PATH, quiet=False)
                    st.success("✅ Download complete.")
                    status.update(label="✅ File ready.", state="complete")
                except Exception as e:
                    st.error(f"❌ Download failed: {e}")
                    status.update(label="❌ Download failed.", state="error")
                    st.stop()
        else:
            st.info(f"📄 Using cached file: `{FILE_PATH}`")

        df = load_and_process_data(FILE_PATH, sample=None)
        visualizations = precompute_visualizations(df)


    st.markdown(f"<h1 style='text-align: center; color: {text_color};'>"
                "<span style='color: darkblue;'>Pro-Israel</span> VS. "
                "<span style='color: green;'>Pro-Palestine</span> Behavior on Social Media</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {text_color};'>Israel-Gaza War Reddit Discussions<br>(OCT 2023 - MAY 2025)</h2>",
                unsafe_allow_html=True)

    pro_israel_score = visualizations.get("_meta").get("pro_israel_score")
    pro_palestine_score = visualizations.get("_meta").get("pro_palestine_score")

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col1:
        st.markdown(f"<div style='background-color: rgba(0, 0, 139, 0.1); padding: 5px; border-radius: 5px; height: 150px;'>"
            f"<p style='font-size: 18px; text-align: center; margin-top: 10px; line-height: 1.2;'>"
            f"<b style='font-size: 22px;'>{pro_israel_score:.2f}</b><br><br>"
            f"<span style='font-size: 16px;'>Avg Comment Score</span><br>"
            f"<span style='font-size: 16px;'>(Likes - Dislikes)</span></p>"
            "</div>", unsafe_allow_html=True)

    with col2:
        pie_fig = visualizations.get("_meta").get('pie')
        st.plotly_chart(pie_fig, use_container_width=True)

    with col3:
        st.markdown(f"<div style='background-color: rgba(0, 128, 0, 0.1); padding: 5px; border-radius: 5px; height: 150px;'>"
            f"<p style='font-size: 18px; text-align: center; margin-top: 10px; line-height: 1.2;'>"
            f"<b style='font-size: 22px;'>{pro_palestine_score:.2f}</b><br><br>"
            f"<span style='font-size: 16px;'>Avg Comment Score</span><br>"
            f"<span style='font-size: 16px;'>(Likes - Dislikes)</span></p>"
            "</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color: {small_text_color}; padding: 5px; border-radius: 5px; margin-bottom: 5px;'>
        <p style='font-size: medium;'>
            <b>ℹ️ Note:</b><br>
            Comments are classified into Pro-Israel and Pro-Palestine groups using a trained classifier. 
            More than 65% of the comments are classified as 'Undefined', meaning their tendency towards a political
            affiliation is not clear. These comments are not shown here. 
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Inject custom CSS for select boxes
    st.markdown(select_box_css, unsafe_allow_html=True)

    # Create the select box for Sub-Topic
    subtopics = [k for k in visualizations.keys() if k != "_meta"]
    selected_subtopic = st.selectbox('Select Topic', subtopics)

    # Create the select box for Feature with a label
    selected_feature = st.selectbox('Select Feature', list(information_hover.keys()))

    # Display the dynamic text box below the select box
    st.markdown(f"""
    <div style='color: {small_text_color}; padding: 5px; border-radius: 5px; margin-top: -10px;'>
        <p style='font-size: medium;'>{information_hover[selected_feature]}</p>
    </div>
    """, unsafe_allow_html=True)

    col1, empty_col, col2 = st.columns([1, 0.05, 1])
    with col1:
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Average {selected_feature} by Topic</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations[selected_subtopic][selected_feature]['radar'], use_container_width=True)

    with col2:
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>{selected_feature} Distribution for Topic: '{selected_subtopic}'</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations[selected_subtopic][selected_feature]['histogram'], use_container_width=True)

    st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Trend of {selected_feature} by Affiliation for Topic '{selected_subtopic}'</h3>", 
                unsafe_allow_html=True)
    st.plotly_chart(visualizations[selected_subtopic][selected_feature]['trend'], use_container_width=True)
    
    st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Comment Volume Trend by Affiliation for Topic '{selected_subtopic}'</h3>", 
                unsafe_allow_html=True)
    st.plotly_chart(visualizations[selected_subtopic][selected_feature]['comment_trend'], use_container_width=True)
    
    st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Factual vs Emotional Speech by Affiliation for Topic '{selected_subtopic}'</h3>",
                unsafe_allow_html=True)
    st.plotly_chart(visualizations[selected_subtopic][selected_feature]['heatmap'], use_container_width=True)

        
if __name__ == "__main__":
    print("📂 Current directory:", os.getcwd())
    print("📄 Files in directory:", os.listdir())
    
    # Now let's go!
    main()
