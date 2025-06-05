# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# Page configuration
st.set_page_config(
    page_title="Billboard Lyrics Analysis",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data with caching
@st.cache_data
def load_data():
    return pd.read_csv('billboard_hot100_cleaned.csv')

df = load_data()

@st.cache_data
def compute_genre_growth(df):
    # Preprocessing
    df = df.copy()
    df['Year'] = df['Year'].astype(int)
    
    # Count songs per genre per year
    genre_yearly_counts = df.groupby(['Year', 'consolidated_genre']).size().reset_index(name='song_count')
    
    # Normalize counts by year
    total_per_year = genre_yearly_counts.groupby('Year')['song_count'].sum().reset_index(name='total_songs')
    genre_yearly_counts = genre_yearly_counts.merge(total_per_year, on='Year')
    genre_yearly_counts['normalized_count'] = genre_yearly_counts['song_count'] / genre_yearly_counts['total_songs']
    
    # Compute average normalized count per decade
    genre_yearly_counts['decade'] = (genre_yearly_counts['Year'] // 10) * 10
    genre_decade_avg = genre_yearly_counts.groupby(['decade', 'consolidated_genre'])['normalized_count'].mean().reset_index()
    
    # Get first and last decades
    first_decade = genre_decade_avg['decade'].min()
    last_decade = genre_decade_avg['decade'].max()
    
    # Compare first vs last decade
    first_avg = genre_decade_avg[genre_decade_avg['decade'] == first_decade]
    last_avg = genre_decade_avg[genre_decade_avg['decade'] == last_decade]
    
    genre_change = pd.merge(first_avg, last_avg, on='consolidated_genre', suffixes=('_first', '_last'))
    genre_change['growth'] = genre_change['normalized_count_last'] - genre_change['normalized_count_first']
    
    # Get top growing and declining genres
    top_growing = genre_change.sort_values('growth', ascending=False).iloc[0]
    top_declining = genre_change.sort_values('growth').iloc[0]
    
    return {
        'top_growing': top_growing,
        'top_declining': top_declining,
        'ts_data': genre_yearly_counts,
        'first_decade': first_decade,
        'last_decade': last_decade
    }

# Add decades column
df['decade'] = (df['Year'] // 10) * 10

# Add song age
current_year = 2024
df['years_ago'] = current_year - df['Year']

# Pre-calculate genre counts
genre_counts = df['consolidated_genre'].value_counts().reset_index()
genre_counts.columns = ['Genre', 'Song Count']

# Custom stopwords for word clouds
extra_stopwords = set(['oh', 'yeah', 'hey', 'la', 'da', 'uh', 'na', 'ha', 'ooh', 'woo', 'hoo', 'chorus', 'verse']) | set(STOPWORDS)

# =========================================
# Sidebar - Filters
# =========================================

st.sidebar.title("Billboard Lyrics Explorer")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Billboard_Hot_100_logo.jpg/320px-Billboard_Hot_100_logo.jpg", width=200)
st.sidebar.markdown("Analyzing 65 years of popular music lyrics (1959-2024)")

# Year range filter
years = st.sidebar.slider(
    "Select Year Range",
    min_value=1959,
    max_value=2024,
    value=(1959, 2024))
st.sidebar.caption(f"Selected: {years[0]} - {years[1]}")

# Genre filter with Select All option
all_genres = df['consolidated_genre'].unique()

# Add a checkbox for "Select All"
select_all_genres = st.sidebar.checkbox("Select All Genres", value=True)

if select_all_genres:
    selected_genres = all_genres
else:
    selected_genres = st.sidebar.multiselect(
        "Select Genres",
        options=all_genres,
        default=all_genres[:5] if len(all_genres) > 5 else all_genres
    )

# Profanity filter
profanity_option = st.sidebar.radio(
    "Profanity Filter:",
    options=["All Songs", "Only Explicit Songs", "No Explicit Songs"],
    index=0
)

# Artist search
artist_query = st.sidebar.text_input("Search Artist:")

# Apply filters
filtered_df = df[
    (df['Year'] >= years[0]) &
    (df['Year'] <= years[1]) &
    (df['consolidated_genre'].isin(selected_genres))
]

if profanity_option == "Only Explicit Songs":
    filtered_df = filtered_df[filtered_df['contains_profanity']]
elif profanity_option == "No Explicit Songs":
    filtered_df = filtered_df[~filtered_df['contains_profanity']]

if artist_query:
    filtered_df = filtered_df[
        filtered_df['Artist'].str.contains(artist_query, case=False)
    ]

# =========================================
# Main Content
# =========================================

st.title("📀 Billboard Hot 100 Lyrics Analysis (1959-2024)")
st.subheader(f"Analyzing {len(filtered_df)} songs from {years[0]} to {years[1]}")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Songs", len(filtered_df))
col2.metric("Artists", filtered_df['Artist'].nunique())
col3.metric("Genres", filtered_df['consolidated_genre'].nunique())
explicit_percent = (filtered_df['contains_profanity'].sum() / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
col4.metric("Explicit Content", f"{explicit_percent:.1f}%")

st.divider()

# =========================================
# Tabs for different analyses
# =========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Genre Trends", "Lyric Analysis", "Profanity Insights", "Artist Spotlight", "Song Explorer"]
)

with tab1:
    # Genre evolution chart 
    st.subheader("Genre Evolution Over Time")
    genre_yearly = filtered_df.groupby(['Year', 'consolidated_genre']).size().unstack().fillna(0)
    fig = px.area(
        genre_yearly,
        title="Genre Popularity by Year",
        labels={'value': 'Number of Songs', 'Year': 'Year'},
        height=500
    )
    fig.update_layout(legend_title_text='Genre')
    st.plotly_chart(fig, use_container_width=True, key="genre_evolution")

    # Dominant genre by decade
    st.subheader("Dominant Genre by Decade")
    
    # Calculate dominant genre for each decade
    decade_genre_counts = filtered_df.groupby(['decade', 'consolidated_genre']).size().reset_index(name='count')
    dominant_genres = decade_genre_counts.loc[decade_genre_counts.groupby('decade')['count'].idxmax()]
    
    # Create visualization
    fig = px.bar(
        dominant_genres,
        x='decade',
        y='count',
        color='consolidated_genre',
        text='consolidated_genre',
        title="Most Popular Genre in Each Decade",
        labels={'count': 'Number of Songs', 'decade': 'Decade'},
        height=500
    )
    
    # Customize appearance
    fig.update_traces(
        textposition='outside',
        marker_line_color='rgb(8,48,107)',
        marker_line_width=1.5,
        opacity=0.7
    )
    fig.update_layout(
        xaxis={'type': 'category'},
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True, key="dominant_genre_decade")

    # Genre insights section
    st.subheader("Genre Insights")
    selected_genre = st.selectbox(
        "Select Genre for Detailed Analysis:",
        options=sorted(filtered_df['consolidated_genre'].unique()),
        key="genre_select_tab1"
    )
    
    # Get data for selected genre
    genre_df = filtered_df[filtered_df['consolidated_genre'] == selected_genre]
    genre_yearly = genre_df.groupby('Year').size().reset_index(name='count')
    total_songs = len(genre_df)
    
    # Calculate decade averages
    genre_df['decade'] = (genre_df['Year'] // 10) * 10
    decade_avg = genre_df.groupby('decade').size().reset_index(name='count')
    decade_avg['avg_songs'] = decade_avg['count'] / (decade_avg['decade'].nunique() * 10)
    
    # Display stats
    st.metric("Total Songs in Selection", total_songs)
    
    # get most common year or fallback if empty
    if not genre_yearly.empty and genre_yearly['count'].notna().any():
        most_common_year = genre_yearly.loc[genre_yearly['count'].idxmax(), 'Year']
    else:
        most_common_year = "N/A"
    
    st.metric("Most Common Year", most_common_year)
    
    # Genre trend chart
    fig = px.line(
        genre_yearly,
        x='Year',
        y='count',
        title=f"{selected_genre} Popularity Trend",
        height=300
    )
    fig.update_traces(line=dict(width=3))
    fig.update_layout(yaxis_title="Number of Songs")
    st.plotly_chart(fig, use_container_width=True, key=f"genre_trend_{selected_genre}")
    
    # Decade averages
    st.markdown("**Average Songs Per Year by Decade**")
    st.dataframe(decade_avg[['decade', 'avg_songs']].set_index('decade'))

    # Top growing/declining genres section
    st.subheader("📈 Top Growing & Declining Genres (1959-2024)")
    
    # Compute genre growth data
    growth_data = compute_genre_growth(df)
    top_growing = growth_data['top_growing']
    top_declining = growth_data['top_declining']
    ts_data = growth_data['ts_data']
    
    # Get time series for top genres
    growing_ts = ts_data[ts_data['consolidated_genre'] == top_growing['consolidated_genre']]
    declining_ts = ts_data[ts_data['consolidated_genre'] == top_declining['consolidated_genre']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Top Growing Genre: {top_growing['consolidated_genre']}**")
        st.caption(f"Growth: {top_growing['growth']:.3f} (from {growth_data['first_decade']}s to {growth_data['last_decade']}s)")
        
        fig = px.area(
            growing_ts,
            x='Year',
            y='normalized_count',
            title=f"Growth Trend: {top_growing['consolidated_genre']}",
            height=400
        )
        fig.update_layout(yaxis_title="Normalized Share")
        st.plotly_chart(fig, use_container_width=True, key="top_growing_genre")
    
    with col2:
        st.markdown(f"**Top Declining Genre: {top_declining['consolidated_genre']}**")
        st.caption(f"Decline: {top_declining['growth']:.3f} (from {growth_data['first_decade']}s to {growth_data['last_decade']}s)")
        
        fig = px.area(
            declining_ts,
            x='Year',
            y='normalized_count',
            title=f"Decline Trend: {top_declining['consolidated_genre']}",
            height=400
        )
        fig.update_layout(yaxis_title="Normalized Share")
        st.plotly_chart(fig, use_container_width=True, key="top_declining_genre")
        
with tab2:
    # Word clouds
    st.subheader("Lyrical Themes by Genre")
    col1, col2 = st.columns(2)

    with col1:
        genre = st.selectbox("Select Genre for Word Cloud:", filtered_df['consolidated_genre'].unique(), key="wordcloud_genre_tab2")

    with col2:
        max_words = st.slider("Max Words in Cloud", 50, 300, 150, key="max_words_tab2")

    genre_df = filtered_df[filtered_df['consolidated_genre'] == genre]
    if not genre_df.empty:
        texts = genre_df['Lyrics_Cleaned'].fillna('').astype(str).tolist()
        text_corpus = " ".join(texts).lower()

        if text_corpus.strip():
            wc = WordCloud(
                width=800,
                height=400,
                background_color='white',
                stopwords=extra_stopwords,
                max_words=max_words,
                colormap='viridis'
            ).generate(text_corpus)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f'Common Words in {genre} Lyrics', fontsize=16)
            st.pyplot(fig)
            st.caption(f"Based on {len(genre_df)} songs")
        else:
            st.warning(f"No lyrics available for {genre}")
    else:
        st.warning(f"No songs found for {genre} with current filters")

    # Word frequency analysis
    st.subheader("Top Words by Genre")
    col1, col2 = st.columns(2)
    
    with col1:
        top_n = st.slider("Number of Top Words to Show", 10, 50, 20, key="top_n_words_tab2")
    
    with col2:
        show_all = st.checkbox("Show All Genres", True, key="show_all_genres")
    
    if show_all:
        genre_df = filtered_df
        title = "Top Words Across All Genres"
    else:
        genre_df = filtered_df[filtered_df['consolidated_genre'] == genre]
        title = f"Top Words in {genre} Lyrics"
    
    if not genre_df.empty:
        all_words = []
        for text in genre_df['Lyrics_Cleaned'].dropna():
            # Clean and split words
            words = re.findall(r'\b\w+\b', text.lower())
            words = [word for word in words if word not in extra_stopwords and len(word) > 1]
            all_words.extend(words)
        
        word_counts = Counter(all_words)
        top_words = word_counts.most_common(top_n)
        
        words_df = pd.DataFrame(top_words, columns=['Word', 'Count'])
        
        fig = px.bar(
            words_df,
            x='Count',
            y='Word',
            orientation='h',
            title=title,
            height=500
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True, key="top_words_chart")
    else:
        st.warning("No lyrics available with current filters")

    # word usage over time
    st.divider()
    st.subheader("🔍 Track Word Usage Over Time")
    
    # Word search input
    search_word = st.text_input("Enter a word to see its usage trend over time:", key="word_search")
    
    if search_word:
        search_word = search_word.strip().lower()
        
        # Create a copy to avoid modifying cached data
        word_df = filtered_df.copy()
        
        # Count songs containing the word (at least once)
        word_df['contains_word'] = word_df['Lyrics_Cleaned'].apply(
            lambda x: 1 if (pd.notna(x) and re.search(rf'\b{re.escape(search_word)}\b', str(x).lower())) else 0
        )
        
        # Filter out years with no lyrics data
        word_trend = word_df[word_df['Lyrics_Cleaned'].notna()].groupby('Year')['contains_word'].agg(['sum', 'count'])
        word_trend = word_trend[word_trend['count'] > 0]  # Remove years with no lyrics data
        
        if not word_trend.empty:
            # Calculate Percentage of songs containing the word
            word_trend['percentage'] = (word_trend['sum'] / word_trend['count']) * 100
            
            # Plot trend
            fig = px.line(
                word_trend.reset_index(),
                x='Year',
                y='percentage',
                title=f"Usage of the word '{search_word}' over time",
                labels={'percentage': 'Percentage of Songs Containing the Word'}
            )
            fig.update_layout(
                yaxis_title="% of Songs Containing Word",
                hovermode='x',
                yaxis=dict(range=[0, min(100, word_trend['percentage'].max() * 1.1)]))
            fig.update_traces(
                hovertemplate="<b>%{x}</b><br>%{y:.2f}% of songs<extra></extra>",
                mode='lines+markers'
            )
            st.plotly_chart(fig, use_container_width=True, key=f"word_trend_{search_word}")
            
            # Show stats
            avg_percentage = word_trend['percentage'].mean()
            peak_year = word_trend['percentage'].idxmax()
            peak_value = word_trend.loc[peak_year, 'percentage']
            
            col1, col2 = st.columns(2)
            col1.metric("Average Usage", f"{avg_percentage:.2f}%")
            col2.metric("Peak Usage", f"{peak_value:.2f}%", f"in {peak_year}")
        else:
            st.warning("No lyrics data available for the selected filters")
    else:
        st.info("Enter a word to see its usage trend over time")

with tab3:
    # Profanity trends
    st.subheader("Explicit Content Trends")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Profanity Over Time**")
        profanity_trend = filtered_df.groupby('Year')['contains_profanity'].mean().reset_index()
        fig = px.line(
            profanity_trend,
            x='Year',
            y='contains_profanity',
            title="Percentage of Songs with Explicit Content",
            height=400
        )
        fig.update_traces(line=dict(color='red', width=3))
        fig.update_layout(yaxis_title="% of Songs with Profanity", yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, key="profanity_trend")

    with col2:
        st.markdown("**Profanity by Genre**")
        profanity_genre = filtered_df.groupby('consolidated_genre')['contains_profanity'].mean().reset_index()
        profanity_genre = profanity_genre.sort_values('contains_profanity', ascending=False)
        fig = px.bar(
            profanity_genre,
            x='consolidated_genre',
            y='contains_profanity',
            color='contains_profanity',
            color_continuous_scale='Reds',
            title="Explicit Content by Genre",
            height=400
        )
        fig.update_layout(xaxis_title="Genre", yaxis_title="% with Profanity", yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True, key="profanity_by_genre")

    # Top explicit songs
    st.subheader("Most Explicit Songs")
    explicit_songs = filtered_df[filtered_df['profanity_count'] > 0].sort_values('profanity_count', ascending=False)
    if not explicit_songs.empty:
        top_explicit = explicit_songs.head(10)[['Year', 'Title', 'Artist', 'consolidated_genre', 'profanity_count']]
        st.dataframe(top_explicit, height=350)
    else:
        st.info("No explicit songs found with current filters")

with tab4:
    # Artist analysis
    st.subheader("Artist Spotlight")
    
    # Top artists by hits
    st.markdown("### Top Artists by Number of Hits")
    top_artists = filtered_df['Artist'].value_counts().head(10).reset_index()
    top_artists.columns = ['Artist', 'Song Count']
    fig = px.bar(
        top_artists,
        x='Artist',
        y='Song Count',
        color='Song Count',
        color_continuous_scale='Blues',
        title="Most Frequent Artists",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True, key="top_artists_chart")
    
    # Artist timeline visualization
    st.markdown("### Artist Timeline Explorer")
    artist = st.selectbox(
        "Select Artist:", 
        filtered_df['Artist'].unique(), 
        key="artist_select_timeline"
    )
    artist_df = filtered_df[filtered_df['Artist'] == artist]
    
    if not artist_df.empty:
        # Fill NaN values in profanity_count with 0
        artist_df = artist_df.copy()
        artist_df['profanity_count'] = artist_df['profanity_count'].fillna(0)
        
        fig = px.scatter(
            artist_df,
            x='Year',
            y='Rank',
            size='profanity_count',
            color='consolidated_genre',
            hover_data=['Title'],
            title=f"{artist}'s Billboard Hits Timeline",
            height=500
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(yaxis_title="Chart Position (Rank)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No songs found for this artist with current filters")
    
    # Artist longevity analysis
    st.markdown("### Artist Longevity Statistics")
    
    # Calculate longevity metrics
    artist_longevity = filtered_df.groupby('Artist').agg(
        first_appearance=('Year', 'min'),
        last_appearance=('Year', 'max'),
        total_hits=('Title', 'count'),
        avg_rank=('Rank', 'mean'),
        explicit_songs=('contains_profanity', 'sum')
    ).reset_index()
    
    artist_longevity['years_active'] = artist_longevity['last_appearance'] - artist_longevity['first_appearance']
    artist_longevity['explicit_percentage'] = (artist_longevity['explicit_songs'] / artist_longevity['total_hits']) * 100
    
    # Filter 
    min_hits = st.slider("Minimum number of hits to include:", 1, 50, 5, key="min_hits_slider")
    longevity_df = artist_longevity[artist_longevity['total_hits'] >= min_hits]
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        options=[
            'years_active (longevity)', 
            'total_hits', 
            'first_appearance',
            'last_appearance',
            'avg_rank'
        ],
        key="longevity_sort"
    )
    
    # Map sort options to columns
    sort_column = {
        'years_active (longevity)': 'years_active',
        'total_hits': 'total_hits',
        'first_appearance': 'first_appearance',
        'last_appearance': 'last_appearance',
        'avg_rank': 'avg_rank'
    }[sort_by]
    
    # Sort the data
    longevity_df = longevity_df.sort_values(sort_column, ascending=sort_by != 'avg_rank')
    
    # Format the table display
    display_columns = {
        'Artist': 'Artist',
        'first_appearance': 'First Year',
        'last_appearance': 'Last Year',
        'years_active': 'Years Active',
        'total_hits': 'Total Hits',
        'avg_rank': 'Avg. Rank',
        'explicit_percentage': '% Explicit'
    }
    
    styled_df = longevity_df[display_columns.keys()].rename(columns=display_columns)
    styled_df['% Explicit'] = styled_df['% Explicit'].round(1)
    styled_df['Avg. Rank'] = styled_df['Avg. Rank'].round(1)
    
    # Display the table
    st.dataframe(
        styled_df.style.format({
            '% Explicit': '{:.1f}%',
            'Avg. Rank': '{:.1f}'
        }).background_gradient(
            subset=['Years Active', 'Total Hits'], 
            cmap='Blues'
        ).highlight_max(
            subset=['Years Active'], 
            color='lightgreen'
        ).highlight_min(
            subset=['Avg. Rank'], 
            color='lightgreen'
        ),
        height=600,
        use_container_width=True
    )

with tab5:
    # Song explorer
    st.subheader("Song Explorer")

    # Search and filter functionality
    col1, col2, col3 = st.columns(3)

    with col1:
        title_filter = st.text_input("Filter by Title:", key="title_filter")

    with col2:
        artist_filter = st.text_input("Filter by Artist:", key="artist_filter")

    with col3:
        genre_filter = st.multiselect(
            "Filter by Genre:",
            options=filtered_df['consolidated_genre'].unique(),
            default=[],
            key="genre_filter"
        )

    # Year range filter
    year_filter = st.slider(
        "Filter by Year Range:",
        min_value=1959,
        max_value=2024,
        value=(1959, 2024),
        key="year_filter"
    )

    # Apply filters
    search_df = filtered_df.copy()
    
    if title_filter:
        search_df = search_df[search_df['Title'].str.contains(title_filter, case=False, na=False)]
    if artist_filter:
        search_df = search_df[search_df['Artist'].str.contains(artist_filter, case=False, na=False)]
    if genre_filter:
        search_df = search_df[search_df['consolidated_genre'].isin(genre_filter)]
    
    search_df = search_df[
        (search_df['Year'] >= year_filter[0]) & 
        (search_df['Year'] <= year_filter[1])
    ]

    if not search_df.empty:
        # Display song list 
        st.dataframe(
            search_df[['Year', 'Title', 'Artist', 'consolidated_genre', 'profanity_count']].sort_values('Year', ascending=False),
            height=400,
            hide_index=True,
            use_container_width=True
        )

        # Song details selection 
        st.subheader("Song Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Artist dropdown
            selected_artist = st.selectbox(
                "Select Artist:",
                options=sorted(search_df['Artist'].unique()),
                key="artist_select_explorer"  # Changed key
            )
            
            # Filter titles based on selected artist
            artist_titles = search_df[search_df['Artist'] == selected_artist]['Title'].unique()
            
        with col2:
            # Title dropdown (filtered by selected artist)
            selected_title = st.selectbox(
                "Select Title:",
                options=sorted(artist_titles),
                key="title_select"
            )

        # Get the selected song data
        song = search_df[
            (search_df['Artist'] == selected_artist) & 
            (search_df['Title'] == selected_title)
        ].iloc[0]

        # Display song details
        st.subheader(f"{song['Title']} by {song['Artist']} ({song['Year']})")

        col1, col2, col3 = st.columns(3)
        col1.metric("Genre", song['consolidated_genre'])
        col2.metric("Chart Position", f"#{song['Rank']}")
        col3.metric("Profanity Count", song['profanity_count'])

        # Lyrics display
        with st.expander("View Lyrics"):
            if pd.isna(song['Lyrics']) or song['Lyrics'].strip() == "":
                st.warning("Lyrics not available for this song")
            else:
                st.text(song['Lyrics'])
    else:
        st.warning("No songs match your search criteria")

# =========================================
# Footer
# =========================================

st.divider()
st.markdown("""
* **Data Source**: Billboard Year-End Hot 100 (1959-2024)
* **Methodology**: Lyrics collected via lyrics.ovh API, genres classified using DeepSeek API
* **Note**: Lyrics not available for all songs (1088 missing)
""")
