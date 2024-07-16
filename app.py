import pandas as pd
import numpy as np
import re
import zipfile
import warnings
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    column_to_name = {
        'Polarity_Sentiment' : 'Polarity Sentiment',
        'Toxicity_Score' : 'Toxicity Score',
        'Belief_Similarity' : 'Belief Speech',
        'Fact_Similarity' : 'Factual Speech',
        'Controversiality' : 'Controversiality'
    } # Built like {feature: column name}
    
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
        hovertemplate=f'Pro-Israel<br>Avg {column_to_name[column]}: %{{r}}<br>Sub Topic: %{{theta}}<extra></extra>'
    ))

    # Add the radar chart for pro-Palestine
    fig.add_trace(go.Scatterpolar(
        r=df_palestine['r'],
        theta=df_palestine['theta'],
        fill='toself',
        name='Pro-Palestine',
        line=dict(color='rgba(0, 128, 0, 0.6)'),
        hovertemplate=f'Pro-Palestine<br>Avg {column_to_name[column]}: %{{r}}<br>Sub Topic: %{{theta}}<extra></extra>'
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
        width=300,  # Set the figure width
        height=300,  # Set the figure height
        margin=dict(t=0, b=15, l=50, r=50)  # Adjusted margins
    )

    return fig


def sentiment_histogram(data, selected_subtopic, column):
    data = data.copy()
    data = data[data['Sub_Topics'].apply(bool)]
    
    # Create subset of the data based on subtopic
    if selected_subtopic != "Overall":
        data = data.explode('Sub_Topics')
        data = data[data['Sub_Topics'] == selected_subtopic]
    
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
            x=0.5, 
            y=1, 
            traceorder='normal',
            font=dict(
                size=12,
                color='#454A4A'
            ),
            bgcolor='rgba(255, 255, 255, 0.5)'
        ),
        xaxis=dict(
            title='Feature Distribution',
            tickfont=dict(color='#454A4A'),  # Change x-axis text color
            titlefont=dict(color='#454A4A')  # Change x-axis title color
        ),
        yaxis=dict(
            title='Percentage of Comments',
            tickfont=dict(color='#454A4A'),  # Change y-axis text color
            titlefont=dict(color='#454A4A'),  # Change y-axis title color
            tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            ticktext=["0%", "10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
        ),
        barmode='group',  # Side-by-side bars
        margin=dict(t=10, b=15, l=50, r=50),  # Adjusted margins
        hoverlabel=dict(font_size=14, font_color='#454A4A'),
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
        xaxis=dict(
            range=[0, 1],
            title='Fact Speaking Score',
            titlefont=dict(color='#454A4A'),  # Change x-axis title color
            tickfont=dict(color='#454A4A'),  # Change x-axis tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        yaxis=dict(
            range=[0, 1],
            title='Belief Speaking Score',
            titlefont=dict(color='#454A4A'),  # Change y-axis title color
            tickfont=dict(color='#454A4A'),  # Change y-axis tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        xaxis2=dict(
            range=[0, 1],
            title='Fact Speaking Score',
            titlefont=dict(color='#454A4A'),  # Change x-axis2 title color
            tickfont=dict(color='#454A4A'),  # Change x-axis2 tick label color
            showgrid=True,
            gridwidth=1,
            gridcolor='#454A4A',
            zeroline=False,
            dtick=0.2
        ),
        yaxis2=dict(
            range=[0, 1],
            title='Belief Speaking Score',
            titlefont=dict(color='#454A4A'),  # Change y-axis2 title color
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
        'Polarity Sentiment': 'Polarity_Sentiment',
        'Toxicity Score': 'Toxicity_Score',
        'Belief Speech': 'Belief_Similarity',
        'Factual Speech': 'Fact_Similarity',
        'Controversiality': 'Controversiality'
    } # Built like {feature: column name}
    visualizations = {} # Create viz for every combination of subtopic adn feature {'subtopic': {'feature': {'heatmap', 'sentiment_histogram', 'radar'}}} 
    radars = {}
    for feature in features_mapper:
        radar_fig = radar(df, features_mapper[feature])
        radars[feature] = radar_fig
    for subtopic in subtopics:
        visualizations[subtopic] = {}
        heatmap_fig = heatmap(df, subtopic)
        for feature in features_mapper:
            radar_fig = radars[feature]
            sentiment_histogram_fig = sentiment_histogram(df, subtopic, features_mapper[feature])
            visualizations[subtopic][feature] = { # Store figs directly as Fig object
                'heatmap': heatmap_fig,
                'sentiment_histogram': sentiment_histogram_fig,
                'radar': radar_fig
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
        'Polarity Sentiment': 'Polarity Sentiment (from TextBlob): This score ranges from -1 (very negative) to 1 (very positive) and represents the sentiment polarity of the text.',
        'Toxicity Score': 'Toxicity Score (from BERT Toxicity): This score ranges from 0 to 1 and indicates the level of toxicity in the comment, with higher scores representing more toxic content.',
        'Belief Speech': 'Belief Speech: This score ranges from 0 to 1 and measures the similarity of the comment to a vector of words representing belief-based speech, calculated using Word2Vec.',
        'Factual Speech': 'Factual Speech: This score ranges from 0 to 1 and measures the similarity of the comment to a vector of words representing factual speech, calculated using Word2Vec.',
        'Controversiality': 'Controversiality: This score is based on Reddit\'s definition and is binary [0, 1], indicating if the comment is controversial within the Reddit community.'
    }

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
        st.markdown(f"<div style='background-color: rgba(0, 0, 139, 0.1); padding: 5px; border-radius: 5px; height: 150px;'>"
            f"<p style='font-size: 18px; text-align: center; margin-top: 10px; line-height: 1.2;'>"
            f"<b style='font-size: 22px;'>{pro_israel_score:.2f}</b><br><br>"
            f"<span style='font-size: 16px;'>Avg Comment Score</span><br>"
            f"<span style='font-size: 16px;'>(Likes - Dislikes)</span></p>"
            "</div>", unsafe_allow_html=True)

    with col2:
        pie_fig = pie_chart(data_dict = {
        'Pro-Israel': 11069,
        'Pro-Palestine': 473671,
        'Unclassified': 43722})
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
            Comments are classified into Pro-Israel and Pro-Palestine groups using a trained SVM model. 
            A comment is considered Pro-X if its probability is 0.6 or higher and at least twice as likely as the other group. 
            About 8.2% of the comments are unclassified and not shown here. 
            An equally partitioned, random sample of the data was used to create this dashboard.
        </p>
    </div>
    """, unsafe_allow_html=True)

    df = balanced_sample(df, 'Affiliation')

    visualizations = precompute_visualizations(df)

    # Inject custom CSS for select boxes
    st.markdown(select_box_css, unsafe_allow_html=True)

    # Create the select box for Sub-Topic
    subtopics = ['Overall'] + df['Sub_Topics'].explode().unique().tolist()
    selected_subtopic = st.selectbox('Select Sub-Topic', subtopics)

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
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Average {selected_feature} by Sub Topic</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations[selected_subtopic][selected_feature]['radar'], use_container_width=True)

    with col2:
        st.markdown(f"<h3 style='text-align: center; color: {text_color};'>{selected_feature} Distribution for Sub Topic: '{selected_subtopic}'</h3>",
                    unsafe_allow_html=True)
        st.plotly_chart(visualizations[selected_subtopic][selected_feature]['sentiment_histogram'], use_container_width=True)

    st.markdown(f"<h3 style='text-align: center; color: {text_color};'>Factual vs Emotional Speech by Affiliation</h3>",
                unsafe_allow_html=True)
    st.plotly_chart(visualizations[selected_subtopic][selected_feature]['heatmap'], use_container_width=True)
 
        
if __name__ == "__main__":
    main()
