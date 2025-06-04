# Billboard Hot 100 Lyrics Analysis (1959-2024)

![Billboard Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Billboard_Hot_100_logo.jpg/320px-Billboard_Hot_100_logo.jpg)

This project analyzes lyrical trends in Billboard Hot 100 songs from 1959 to 2024 through a comprehensive data pipeline covering collection, preprocessing, enrichment, and analysis.

## 🎯 Project Overview

I've developed an automated pipeline to:
1. Collect Billboard chart data from Wikipedia
2. Fetch song lyrics using async API requests
3. Clean and preprocess text data
4. Classify song genres using AI
5. Detect profanity in lyrics
6. Prepare data for sentiment and thematic analysis

The final dataset contains **6,500 songs** across 65 years of popular music.

## 🔑 Key Features

### 📊 Data Collection

#### 📜 Data Collection Workflow
```mermaid
graph TD
    A[Scrape Wikipedia] --> B[Get Billboard Data]
    B --> C[Check Lyrics Cache]
    C --> D{Has Lyrics?}
    D -->|Yes| E[Use Cached Lyrics]
    D -->|No| F[Fetch via API]
    F --> G[Retry with Title/Artist Variations]
    G --> H[Save to Cache]
    H --> I[Save CSV]
    I --> J[Zip Yearly Files]
```

- Scraped Billboard Year-End Hot 100 lists (1959-2024) from Wikipedia
- Implemented intelligent async lyrics fetching from lyrics.ovh API
- Built caching system to avoid redundant API calls
- Created retry logic with artist/title variations
- Zipped yearly CSV files for efficient storage

### 🧹 Data Cleaning
- Consolidated 65 yearly CSV files into unified dataset
- Standardized artist names (handling "feat." variations)
- Processed lyrics with:
  - Structure marker removal ([Verse], [Chorus])
  - Contraction expansion ("don't" → "do not")
  - Special character removal
  - Case normalization
  - Stopword removal
  - Lemmatization

### 🎸 Genre Classification
- Used DeepSeek API for song-level genre prediction
- Consolidated raw genres into 12 main categories:
  - Pop, Rock, R&B, Hip-Hop, Country
  - Dance/Electronic, Jazz, Blues, Soul
  - Gospel/Christian, Classical, Reggae

### ⚠️ Profanity Detection
- Combined two authoritative profanity word lists:
  1. [zacanger/profane-words](https://github.com/zacanger/profane-words)
  2. [LDNOOBW List](https://github.com/LDNOOBW)
- Implemented fuzzy matching for obfuscated profanity:
  - "f@ck", "sh1t", "b!tch"
- Added annotations:
  - `profanity_words`: List of detected words
  - `profanity_count`: Numeric count
  - `contains_profanity`: Boolean flag

## 📂 Data Output

The final dataset (`billboard_hot100_cleaned.csv`) contains:

| Column | Description | Example |
|--------|-------------|---------|
| `Rank` | Billboard position | 1 |
| `Title` | Song title | "Blinding Lights" |
| `Artist` | Standardized artist name | "The Weeknd" |
| `Year` | Chart year | 2020 |
| `Lyrics` | Raw lyrics text | "I been tryna call..." |
| `Lyrics_Cleaned` | Processed lyrics | "tryna call..." |
| `genre` | DeepSeek-predicted genre | "synth-pop" |
| `consolidated_genre` | Standardized genre | "Pop" |
| `profanity_words` | List of profane terms | ["fuck"] |
| `profanity_count` | Profanity frequency | 4 |
| `contains_profanity` | Profanity flag | True |

## 📈 Next Steps for Analysis

The cleaned dataset is ready for:
1. **Sentiment Analysis**: Track emotional valence over decades
2. **Topic Modeling**: Identify recurring themes in lyrics
3. **Word Frequency**: Compare vocabulary across genres/eras
4. **Profanity Trends**: Analyze explicitness over time
5. **Genre Evolution**: Visualize genre popularity shifts
