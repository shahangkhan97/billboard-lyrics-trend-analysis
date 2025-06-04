# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

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
    df = pd.read_csv('billboard_hot100_cleaned.csv')
    
    # Fill missing profanity_count values
    df['profanity_count'] = df['profanity_count'].fillna(0)
    
    # Add decades column
    df['decade'] = (df['Year'] // 10) * 10
    
    return df

df = load_data()

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

# Genre filter - with "All" option
all_genres = df['consolidated_genre'].unique()
select_all = st.sidebar.checkbox("Select All Genres", True)
if select_all:
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

# Artist search (including featured artists)
artist_query = st.sidebar.text_input("Search Artist (including featured):")
if artist_query:
    # Search for artist in main artist or featured
    artist_query = artist_query.lower()
    artist_mask = df['Artist'].str.lower().apply(
        lambda x: artist_query in x or any(artist_query in feat.lower() for feat in x.split(' Feat. '))
    )
else:
    artist_mask = pd.Series([True] * len(df))

# Apply filters
filtered_df = df[
    (df['Year'] >= years[0]) & 
    (df['Year'] <= years[1]) &
    (df['consolidated_genre'].isin(selected_genres)) &
    artist_mask
]

if profanity_option == "Only Explicit Songs":
    filtered_df = filtered_df[filtered_df['contains_profanity']]
elif profanity_option == "No Explicit Songs":
    filtered_df = filtered_df[~filtered_df['contains_profanity']]

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
    # Genre overview
    st.subheader("Genre Distribution (All 6500 Songs)")
    genre_counts = df['consolidated_genre'].value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Genre Counts**")
        st.dataframe(genre_counts, height=400)
    
    with col2:
        st.markdown("**Genre Distribution**")
        fig = px.pie(
            genre_counts, 
            names='Genre', 
            values='Count',
            hole=0.3,
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
    
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
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre growth/decline analysis
    st.subheader("Genre Growth & Decline (1959 vs 2024)")
    
    # Calculate change from first to last period
    start_period = df[df['Year'].between(1959, 1969)]
    end_period = df[df['Year'].between(2015, 2024)]
    
    start_counts = start_period['consolidated_genre'].value_counts().reset_index()
    start_counts.columns = ['Genre', 'Start_Count']
    
    end_counts = end_period['consolidated_genre'].value_counts().reset_index()
    end_counts.columns = ['Genre', 'End_Count']
    
    genre_change = pd.merge(start_counts, end_counts, on='Genre', how='outer').fillna(0)
    genre_change['Change'] = genre_change['End_Count'] - genre_change['Start_Count']
    genre_change['Pct_Change'] = (genre_change['Change'] / genre_change['Start_Count']) * 100
    genre_change = genre_change.sort_values('Pct_Change', ascending=False)
    
    # Top growing genres
    st.markdown("**Top Growing Genres**")
    top_growing = genre_change.nlargest(5, 'Pct_Change')
    fig = px.bar(
        top_growing,
        x='Genre',
        y='Pct_Change',
        color='Pct_Change',
        color_continuous_scale='Greens',
        text='Pct_Change',
        height=400
    )
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    fig.update_layout(yaxis_title="Percentage Change")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top declining genres
    st.markdown("**Top Declining Genres**")
    top_declining = genre_change.nsmallest(5, 'Pct_Change')
    fig = px.bar(
        top_declining,
        x='Genre',
        y='Pct_Change',
        color='Pct_Change',
        color_continuous_scale='Reds',
        text='Pct_Change',
        height=400
    )
    fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
    fig.update_layout(yaxis_title="Percentage Change")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    # Word clouds
    st.subheader("Lyrical Themes by Genre")
    col1, col2 = st.columns(2)
    
    with col1:
        genre = st.selectbox("Select Genre for Word Cloud:", filtered_df['consolidated_genre'].unique())
    
    with col2:
        max_words = st.slider("Max Words in Cloud", 50, 300, 150)
    
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
        top_n = st.slider("Number of Top Words to Show", 10, 50, 20)
    
    with col2:
        show_all = st.checkbox("Show All Genres", True)
    
    if show_all:
        genre_df = filtered_df
        title = "Top Words Across All Genres"
    else:
        genre_df = filtered_df[filtered_df['consolidated_genre'] == genre]
        title = f"Top Words in {genre} Lyrics"
    
    if not genre_df.empty:
        all_words = []
        for text in genre_df['Lyrics_Cleaned'].dropna():
            words = text.split()
            words = [word for word in words if word not in extra_stopwords]
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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No lyrics available with current filters")

    # Word search section
st.subheader("Word Usage Analysis")
search_word = st.text_input("Enter a word to analyze:")

if search_word:
    # Add word count columns
    df['word_count'] = df['lyrics'].apply(lambda x: count_word_occurrences(x, search_word))
    df['contains_word'] = df['word_count'] > 0
    
    # Filter songs containing the word
    word_songs = df[df['contains_word']].copy()
    
    # Time trend visualization
    yearly = df.groupby('year').agg(
        total_songs=('track_name', 'count'),
        songs_with_word=('contains_word', 'sum'),
        total_occurrences=('word_count', 'sum')
    ).reset_index()
    
    yearly['frequency_percent'] = yearly['songs_with_word'] / yearly['total_songs'] * 100
    
    fig = px.line(yearly, x='year', y='frequency_percent',
                  title=f'Usage of "{search_word}" Over Time',
                  labels={'frequency_percent': '% of Songs Containing Word'})
    st.plotly_chart(fig)
    
    # Display songs table with filtering
    st.subheader(f"Songs Containing '{search_word}'")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        year_range = st.slider("Year Range", 
                              min_value=int(df['year'].min()),
                              max_value=int(df['year'].max()),
                              value=(1950, 2020))
    with col2:
        min_occurrences = st.slider("Minimum Occurrences", 1, 10, 1)
    
    # Apply filters
    filtered_songs = word_songs[
        (word_songs['year'] >= year_range[0]) &
        (word_songs['year'] <= year_range[1]) &
        (word_songs['word_count'] >= min_occurrences)
    ]
    
    # Show results
    st.dataframe(filtered_songs[['track_name', 'artist_name', 'year', 'word_count']])
    
    # Show lyrics snippets
    st.subheader("Lyrics Snippets")
    for _, row in filtered_songs.iterrows():
        lyrics = row['lyrics']
        matches = re.finditer(rf'\b{re.escape(search_word)}\b', lyrics, re.IGNORECASE)
        positions = [match.start() for match in matches]
        
        for pos in positions:
            start = max(0, pos - 30)
            end = min(len(lyrics), pos + len(search_word) + 30)
            snippet = lyrics[start:end].replace('\n', ' ')
            st.write(f"**{row['track_name']}** ({row['year']}): ...{snippet}...")

with tab3:
    # Profanity trends
    st.subheader("Explicit Content Trends")
    
    # Profanity over time
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
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Artist longevity and consistency
    artist_stats = filtered_df.groupby('Artist').agg(
        first_year=('Year', 'min'),
        last_year=('Year', 'max'),
        song_count=('Title', 'count'),
        avg_rank=('Rank', 'mean')
    ).reset_index()
    
    artist_stats['career_span'] = artist_stats['last_year'] - artist_stats['first_year'] + 1
    artist_stats['consistency'] = artist_stats['song_count'] / artist_stats['career_span']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Consistent Artists**")
        st.caption("Highest average songs per active year")
        consistent_artists = artist_stats.nlargest(10, 'consistency')[['Artist', 'consistency', 'song_count', 'career_span']]
        consistent_artists['consistency'] = consistent_artists['consistency'].round(1)
        st.dataframe(consistent_artists, height=400)
    
    with col2:
        st.markdown("**Longest Career Spans**")
        st.caption("Artists with the most years between first and last hit")
        long_artists = artist_stats.nlargest(10, 'career_span')[['Artist', 'career_span', 'first_year', 'last_year']]
        st.dataframe(long_artists, height=400)
    
    # Artist timeline
    st.subheader("Artist Timeline")
    artist = st.selectbox("Select Artist:", filtered_df['Artist'].unique())
    artist_df = filtered_df[filtered_df['Artist'] == artist]
    
    if not artist_df.empty:
        # Fill missing profanity counts
        artist_df = artist_df.copy()
        artist_df['profanity_count'] = artist_df['profanity_count'].fillna(0)
        
        fig = px.scatter(
            artist_df, 
            x='Year', 
            y='Rank',
            size='profanity_count',
            color='consolidated_genre',
            hover_data=['Title'],
            title=f"{artist}'s Billboard Hits",
            height=500
        )
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(yaxis_title="Chart Position (Rank)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No songs found for this artist with current filters")

with tab5:
    # Song explorer
    st.subheader("Song Explorer")
    
    # Search functionality
    col1, col2 = st.columns(2)
    
    with col1:
        search_term = st.text_input("Search song titles:")
    
    with col2:
        year_filter = st.slider("Filter by year", 1959, 2024, (1959, 2024))
    
    # Apply search filters
    search_df = filtered_df.copy()
    if search_term:
        search_df = search_df[search_df['Title'].str.contains(search_term, case=False)]
    search_df = search_df[(search_df['Year'] >= year_filter[0]) & (search_df['Year'] <= year_filter[1])]
    
    if not search_df.empty:
        # Create display names for dropdown
        search_df['display_name'] = search_df['Title'] + " by " + search_df['Artist'] + " (" + search_df['Year'].astype(str) + ")"
        
        # Display song list
        st.dataframe(
            search_df[['Year', 'Title', 'Artist', 'consolidated_genre', 'profanity_count']].sort_values('Year', ascending=False),
            height=400,
            hide_index=True
        )
        
        # Song details selection
        selected_display = st.selectbox("Select a song for details:", search_df['display_name'].unique())
        
        # Get the selected song
        song = search_df[search_df['display_name'] == selected_display].iloc[0]
        
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
**Data Source**: Billboard Year-End Hot 100 (1959-2024)  
**Methodology**: Lyrics collected via lyrics.ovh API, genres classified using DeepSeek API  
**Note**: Lyrics not available for all songs (1088 missing)  
""")
