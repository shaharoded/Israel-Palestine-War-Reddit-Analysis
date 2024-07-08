import pandas as pd
import numpy as np
import re
import zipfile
import warnings
import io
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from PIL import Image

# Ignore PerformanceWarning
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)

# Google Drive .zip file ID
FILE_NAME = 'Preprocessed_Dataset.csv'
ZIP_FILE_PATH = 'Preprocessed_Dataset_Sample.zip'

# Function to parse the Sub_Topics string
def parse_subtopics(subtopics_str):
    '''
    Fix subtopics column to a workable format.
    '''
    if isinstance(subtopics_str, str):
        subtopics = re.sub(r'[{}]', '', subtopics_str).split(', ')
        subtopics = subtopics = [subtopic.strip("'").capitalize() for subtopic in subtopics]
        return set(subtopics) if subtopics else set()
    return set()

def balanced_sample(df, column):
    '''
    Create a balanced sample of comments from both groups
    '''
    # Count the number of occurrences in each group
    group_counts = df[column].value_counts()

    # Identify the minority group and its count
    minority_group = group_counts.idxmin()
    minority_count = group_counts.min()

    # Sample the same amount from the majority group
    majority_group = group_counts.idxmax()
    majority_sample = df[df[column] == majority_group].sample(n=minority_count, random_state=42)

    # Get all rows from the minority group
    minority_sample = df[df[column] == minority_group]

    # Concatenate the minority and majority samples
    balanced_df = pd.concat([minority_sample, majority_sample], ignore_index=True)

    return balanced_df

def radar(data, column):
    # Preprocess the data
    data = data.copy()
    data = data[data['Sub_Topics'].apply(bool)]

    # Explode the data to have one row per subtopic
    exploded_data = data.explode('Sub_Topics')
    exploded_data = exploded_data.dropna(subset=['Sub_Topics'])

    # Group by Affiliation and Subtopics to calculate the average sentiment
    grouped_data = exploded_data.groupby(['Affiliation', 'Sub_Topics']).agg({column: 'mean'}).reset_index()

    # Prepare data for radar plot by group
    pro_israel_data = grouped_data[grouped_data['Affiliation'] == 'Pro-Israel']
    pro_palestine_data = grouped_data[grouped_data['Affiliation'] == 'Pro-Palestine']

    # Ensure Subtopics are aligned between the two groups for consistent radar plot structure
    all_subtopics = set(pro_israel_data['Sub_Topics']).union(set(pro_palestine_data['Sub_Topics']))
    for subtopic in all_subtopics:
        if subtopic not in pro_israel_data['Sub_Topics'].values:
            pro_israel_data = pro_israel_data.append({'Affiliation': 'Pro-Israel', 'Sub_Topics': subtopic, column: 0}, ignore_index=True)
        if subtopic not in pro_palestine_data['Sub_Topics'].values:
            pro_palestine_data = pro_palestine_data.append({'Affiliation': 'Pro-Palestine', 'Sub_Topics': subtopic, column: 0}, ignore_index=True)

    # Sort by Subtopics to ensure consistency
    pro_israel_data = pro_israel_data.sort_values(by='Sub_Topics')
    pro_palestine_data = pro_palestine_data.sort_values(by='Sub_Topics')
    subtopics_israel = pro_israel_data['Sub_Topics'].tolist()
    values_israel = pro_israel_data[column].tolist()
    subtopics_palestine = pro_palestine_data['Sub_Topics'].tolist()
    values_palestine = pro_palestine_data[column].tolist()

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
        hovertemplate=f'Pro-Israel<br>Avg_{column}: %{{r}}<br>SubTopic: %{{theta}}<extra></extra>'
    ))

    # Add the radar chart for pro-Palestine
    fig.add_trace(go.Scatterpolar(
        r=df_palestine['r'],
        theta=df_palestine['theta'],
        fill='toself',
        name='Pro-Palestine',
        line=dict(color='rgba(0, 128, 0, 0.6)'),
        hovertemplate=f'Pro-Palestine<br>Avg_{column}: %{{r}}<br>SubTopic: %{{theta}}<extra></extra>'
    ))

    # Update layout for title and axis labels
    fig.update_layout(
        showlegend=False,
        polar=dict(
            radialaxis=dict(visible=True, range=[min(values_israel + values_palestine), max(values_israel + values_palestine)]),
            angularaxis=dict(
                tickfont=dict(size=10),
                categoryarray=subtopics_israel + [subtopics_israel[0]],  # Set custom category order
                categoryorder='array'
            )
        ),
        width=300,  # Set the figure width
        height=300,  # Set the figure height
        margin=dict(t=0, b=15, l=50, r=50)  # Adjusted margins
    )

    return fig


def sentiment_histogram(df, selected_subtopic):
    data = df.copy()

    if selected_subtopic != "Overall":
        data = data.explode('Sub_Topics')
        data = data[data['Sub_Topics'] == selected_subtopic]

    # Separate data for Pro-Israel and Pro-Palestine
    pro_israel_df = data[data['Affiliation'] == 'Pro-Israel']
    pro_palestine_df = data[data['Affiliation'] == 'Pro-Palestine']

    # Create a figure
    fig = make_subplots(rows=1, cols=1)

    # Define bins for sentiment scores
    bins = np.arange(-1, 1.2, 0.2)  # Bins from -1 to 1 with step 0.2

    # Get data for the selected subtopic
    pro_israel_data = pro_israel_df['Polarity_Sentiment']
    pro_palestine_data = pro_palestine_df['Polarity_Sentiment']

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
    hover_text_pro_israel = [f"Sentiment: {bins[i]:.1f} to {bins[i+1]:.1f},\n{perc_pro_israel[i]:.1f}% of the group's comments" for i in range(len(perc_pro_israel))]
    hover_text_pro_palestine = [f"Sentiment: {bins[i]:.1f} to {bins[i+1]:.1f},\n{perc_pro_palestine[i]:.1f}% of the group's comments" for i in range(len(perc_pro_palestine))]

    # Create traces
    trace_israel = go.Bar(
        x=bin_centers,
        y=perc_pro_israel,
        name='Pro-Israel',
        marker_color='rgba(0, 0, 255, 0.6)',
        opacity=0.7,
        width=0.1,
        hovertext=hover_text_pro_israel,
        hoverinfo='text'
    )
    trace_palestine = go.Bar(
        x=bin_centers,
        y=perc_pro_palestine,
        name='Pro-Palestine',
        marker_color='rgba(0, 128, 0, 0.6)',
        opacity=0.7,
        width=0.1,
        hovertext=hover_text_pro_palestine,
        hoverinfo='text'
    )

    fig.add_traces([trace_israel, trace_palestine])

    # Update layout
    fig.update_layout(
        showlegend=True,
        legend=dict(
            x=0.3, 
            y=0.7, 
            traceorder='normal',
            font=dict(
                size=12,
                color='#2f2f2f'
            ),
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        xaxis_title='Polarity Sentiment',
        yaxis_title='Percentage of Comments',
        barmode='group',  # Side-by-side bars
        yaxis=dict(tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], ticktext=["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]),
        margin=dict(t=10, b=15, l=50, r=50),  # Adjusted margins
        height=300
    )

    return fig


def heatmap(df, subtopic):
    data = df.copy()

    if subtopic != "Overall":
        data = data.explode('Sub_Topics')
        data = data[data['Sub_Topics'] == subtopic]

    # Normalize Fact_Similarity and Belief_Similarity to be between 0 and 1
    data['Fact_Similarity'] = (data['Fact_Similarity'] - data['Fact_Similarity'].min()) / (data['Fact_Similarity'].max() - data['Fact_Similarity'].min())
    data['Belief_Similarity'] = (data['Belief_Similarity'] - data['Belief_Similarity'].min()) / (data['Belief_Similarity'].max() - data['Belief_Similarity'].min())

    # Function to create hexbin traces
    def create_hexbin_trace(df, affiliation, color):
        subset = df[df['Affiliation'] == affiliation]

        x = subset['Fact_Similarity']
        y = subset['Belief_Similarity']

        hist, xedges, yedges = np.histogram2d(x, y, bins=[20, 20], range=[[0, 1], [0, 1]])
        hist = hist.T
        hist_percentile = (hist / hist.max()) * 100  # Normalize to 100%
        hist_percentage = (hist / hist.sum()) * 100  # Percentage of total samples in each bin

        trace = go.Heatmap(
            x=xedges,
            y=yedges,
            z=hist_percentile,
            colorscale=color,
            showscale=False,
            name=f'{affiliation}',
            hovertemplate=(
                'Fact Speaking Score: %{x}<br>'
                'Belief Speaking Score: %{y}<br>'
                'Density Measure (Percentile): %{z:.2f}%<br>'
                'Percent of Group: %{customdata:.2f}%<extra></extra>'
            ),
            customdata=hist_percentage  # Add custom data for the hover info
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
        xaxis=dict(range=[0, 1], title='Fact Speaking Score', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2),
        yaxis=dict(range=[0, 1], title='Belief Speaking Score', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2),
        xaxis2=dict(range=[0, 1], title='Fact Speaking Score', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2),
        yaxis2=dict(range=[0, 1], title='Belief Speaking Score', showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2),
        showlegend=True,
        margin=dict(l=0, r=0, t=40, b=80),
        plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to be transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set paper background to be transparent
        font=dict(color='black')
    )

    # Ensure the grid is visible and fits properly
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgrey', zeroline=False, dtick=0.2)

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
                font=dict(size=14, color="darkgrey"),
                showarrow=False,
                xanchor='center'
            )
        ],
        showlegend=False,  # Remove legend
        height=150,  # Adjust height
        margin=dict(l=10, r=10, t=30, b=10)  # Adjust margins
    )
    
    return fig


# def wordcloud(df, subtopic, colormap):
#     data = df.copy()

#     if subtopic != "Overall":
#         data = data.explode('Sub_Topics')
#         data = data[data['Sub_Topics'] == subtopic]
        
#     text = ' '.join(data['Normalized_English_Comment'].dropna())

#     if not text:
#         st.error("No text data available to generate wordcloud.")
#         return None
    
#     try:
#         # Generate word cloud
#         wordcloud = WordCloud(width=1200, height=1000, background_color='white', colormap=colormap).generate(text)
        
#         # Create a figure and display the word cloud
#         plt.figure(figsize=(12, 10))
#         plt.imshow(wordcloud, interpolation='bilinear')
#         plt.axis('off')
        
#         # Save the figure to a buffer
#         buf = io.BytesIO()
#         plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.05, dpi=300)
#         buf.seek(0)
#         plt.close()
        
#         return buf
#     except Exception as e:
#         st.error(f"Error generating wordcloud: {e}")
#         return None


@st.cache_data
def load_and_process_data(zip_filepath, csv_filename):
    '''
    Pre-process the data, and cache to save calculations.
    '''
    try:
        # Extract the zip file in memory
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            with zip_ref.open(csv_filename) as f:
                df = pd.read_csv(f, index_col=None, on_bad_lines='skip')

        # Process the DataFrame
        df['Sub_Topics'] = df['Sub_Topics'].apply(parse_subtopics)
        valid_affiliations = {'Pro-Israel', 'Pro-Palestine'}
        df = df[df['Affiliation'].isin(valid_affiliations)]
        df['Score'] = pd.to_numeric(df['Score'], errors='coerce')
        df = df.dropna(how='any').reset_index(drop=True)
        df = df[df['Sub_Topics'].apply(lambda x: x != set())].reset_index(drop=True)

        return df

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error


@st.cache_resource
def precompute_visualizations(df):
    '''
    Pre-compute all visualizations to avoid heavy calculation for every filter change.
    '''
    subtopics = ['Overall'] + df['Sub_Topics'].explode().unique().tolist()
    features_mapper = {
        'Toxicity Score': 'Toxicity_Score',
        'Polarity Sentiment': 'Polarity_Sentiment',
        'Belief Speech': 'Belief_Similarity',
        'Factual Speech': 'Fact_Similarity',
        'Comment Score': 'Score',
        'Controversiality Ratio': 'Controversiality'
    } # Built like {feature: column name}
    visualizations = {'by_subtopic': {}, 'by_feature': {}}
    for feature in features_mapper:
        radar_fig = radar(df, features_mapper[feature])
        visualizations['by_feature'][feature] = radar_fig
    for subtopic in subtopics:
        heatmap_fig = heatmap(df, subtopic)
        sentiment_histogram_fig = sentiment_histogram(df, subtopic)
        visualizations['by_subtopic'][subtopic] = {
            'heatmap': heatmap_fig,
            'sentiment_histogram': sentiment_histogram_fig
        }
    return visualizations


def main():
    text_color = 'darkgrey'
    dark_text_color = '#2f2f2f'
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
    
    st.markdown(f"<h1 style='text-align: center; color: {text_color};'>"
                "<span style='color: darkblue;'>Pro-Israel</span> VS. "
                "<span style='color: green;'>Pro-Palestine</span> Behavior on Social Media</h1>",
                unsafe_allow_html=True)
    st.markdown(f"<h2 style='text-align: center; color: {text_color};'>Regarding the Israel-Gaza War (2023-2024)</h2>",
                unsafe_allow_html=True)

    df = load_and_process_data(ZIP_FILE_PATH, FILE_NAME)

    pro_israel_score = df[df['Affiliation'] == 'Pro-Israel']['Score'].mean()
    pro_palestine_score = df[df['Affiliation'] == 'Pro-Palestine']['Score'].mean()

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col1:
        st.markdown(f"<div style='background-color: rgba(0, 0, 139, 0.1); padding: 10px; border-radius: 5px; height: 150px;'>"
                    f"<p style='font-size: large; text-align: center; margin-top: 30px;'>"
                    f"<b>{pro_israel_score:.2f}</b><br><span style='font-size: small;'>Avg Comment Score (Likes - Dislikes)</span></p>"
                    "</div>", unsafe_allow_html=True)

    with col2:
        pie_fig = pie_chart(data_dict = {
        'Pro-Israel': 11069,
        'Pro-Palestine': 473671,
        'Unclassified': 43722})
        st.plotly_chart(pie_fig, use_container_width=True)

    with col3:
        st.markdown(f"<div style='background-color: rgba(0, 128, 0, 0.1); padding: 10px; border-radius: 5px; height: 150px;'>"
                    f"<p style='font-size: large; text-align: center; margin-top: 30px;'>"
                    f"<b>{pro_palestine_score:.2f}</b><br><span style='font-size: small;'>Avg Comment Score (Likes - Dislikes)</span></p>"
                    "</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='color: {dark_text_color}; padding: 10px; border-radius: 5px;'>
        <p style='font-size: small;'>
            <b>ℹ️ Note:</b><br>
            Comments are classified into Pro-Israel and Pro-Palestine groups using a trained SVM model. 
            A comment is considered Pro-X if its probability is 0.6 or higher and at least twice as likely as the other group. 
            About 8.2% of the comments are unclassified and not shown here. 
            An equally partitioned, random sample of the data was used to create this dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = balanced_sample(df, 'Affiliation')

    visualizations = precompute_visualizations(df)

    # create SelectBox for SubTopic
    # Inject custom CSS
    st.markdown(select_box_css, unsafe_allow_html=True)
    subtopics = ['Overall'] + df['Sub_Topics'].explode().unique().tolist()
    selected_subtopic = st.selectbox('Select Subtopic', subtopics)
    
    # create SelectBox for Feature
    # Inject custom CSS
    st.markdown(select_box_css, unsafe_allow_html=True)
    features = ['Toxicity Score', 'Polarity Sentiment', 'Belief Speech',
        'Factual Speech', 'Comment Score', 'Controversiality Ratio']
    selected_feature = st.selectbox('Select Feature', features)

    col1, empty_col, col2 = st.columns([1, 0.05, 1])
    with col1:
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Average {selected_feature} by SubTopic</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations['by_feature'][selected_feature], use_container_width=True)

    with col2:
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Sentiment Distribution by SubTopic</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations['by_subtopic'][selected_subtopic]['sentiment_histogram'], use_container_width=True)

    st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Factual vs Emotional Speech by Affiliation</h3>",
                unsafe_allow_html=True)
    st.plotly_chart(visualizations['by_subtopic'][selected_subtopic]['heatmap'], use_container_width=True)

    # st.markdown(f"<h3 style='text-align: center; color: {text_color};'>WordClouds</h3>",
    #             unsafe_allow_html=True)
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.image(Image.open(visualizations['by_subtopic'][selected_subtopic]['pro_israel_wordcloud']), use_column_width=True)

    # with col2:
    #     st.image(Image.open(visualizations['by_subtopic'][selected_subtopic]['pro_palestine_wordcloud']), use_column_width=True)

        
if __name__ == "__main__":
    main()