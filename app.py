
import streamlit as st
import pandas as pd
import plotly.express as px
from collections import defaultdict
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Page Config
st.set_page_config(page_title="GameSense :D", layout="wide")

# Load CSS 
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load Data & ML Assets 
df = pd.read_csv("assets/steam_processed.csv")
tfidf = pickle.load(open("assets/tfidf.pkl", "rb"))
rating_model = pickle.load(open("assets/rating_model.pkl", "rb"))
mlb = pickle.load(open("assets/mlb.pkl", "rb"))

# Ensure popularity exists
if 'popularity' not in df.columns:
    df['popularity'] = df['positive_ratings'] + df['negative_ratings']

# Header 
st.markdown("<div class='floating-game'>🎮</div>", unsafe_allow_html=True)
st.markdown("<h1 class='gradient-title'>🎮 GameSense :D</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Discover the best games using smart filters, ratings, and insights.</p>", unsafe_allow_html=True)

# Sidebar Filters 
st.sidebar.header("Filters")
st.sidebar.markdown("<div class='sidebar-float'>🎮</div>", unsafe_allow_html=True)

search = st.sidebar.text_input("🔍 Search Game")
min_rating = st.sidebar.slider("⭐ Minimum Rating", 0.0, 5.0, 2.5)
max_price = st.sidebar.slider("💰 Maximum Price", 0.0, float(df['price'].max()), 50.0)

# Multi-select Genres
all_genres = df['genres'].dropna().str.split(';').sum()
unique_genres = sorted(set(all_genres))
selected_genres = st.sidebar.multiselect("🎲 Genres", unique_genres, default=None)

# Multi-select Platforms
all_platforms = df['platforms'].dropna().str.split(';').sum()
unique_platforms = sorted(set(all_platforms))
selected_platforms = st.sidebar.multiselect("🖥️ Platforms", unique_platforms, default=None)

# Developer Panel
st.sidebar.header("🛠️ Developer Panel")
st.sidebar.write("Total Games:", len(df))
st.sidebar.write("Unique Genres:", len(unique_genres))
st.sidebar.write("Unique Platforms:", len(unique_platforms))
st.sidebar.write("Max Price:", df['price'].max())
st.sidebar.write("Average Rating:", round(df['rating'].mean(),2))

# Filtering Logic
filtered_df = df[(df['rating'] >= min_rating) & (df['price'] <= max_price)]

if selected_genres:
    filtered_df = filtered_df[filtered_df['genres'].apply(lambda x: any(g in x.split(';') for g in selected_genres))]

if selected_platforms:
    filtered_df = filtered_df[filtered_df['platforms'].apply(lambda x: any(p in x.split(';') for p in selected_platforms))]

if search:
    filtered_df = filtered_df[filtered_df['name'].str.contains(search, case=False)]

#  Similar Games
st.sidebar.subheader("🤖 Recommendations")
selected_game = st.sidebar.text_input("Game to get similar games")
if selected_game:
    matches = df[df['name'].str.lower().str.contains(selected_game.lower())]
    if not matches.empty:
        idx = matches.index[0]
        selected_vec = tfidf.transform([df.loc[idx, 'features']])
        sim_scores = cosine_similarity(selected_vec, tfidf.transform(df['features'])).flatten()
        top_idx = sim_scores.argsort()[::-1][1:6]  # top 5 similar
        rec_games = df.iloc[top_idx][['name','rating','price']]
        st.sidebar.write("Similar Games:")
        st.sidebar.dataframe(rec_games)
    else:
        st.sidebar.write("Game not found!")

# Predicted Ratings 
st.sidebar.subheader("🔮 Predicted Ratings")
filtered_ml = filtered_df.copy()
filtered_ml['genres_list'] = filtered_ml['genres'].fillna('').str.split(';')
genres_encoded = mlb.transform(filtered_ml['genres_list'])
X_new = pd.DataFrame(genres_encoded, columns=mlb.classes_)
X_new['price'] = filtered_ml['price'].values
X_new['popularity'] = filtered_ml['popularity'].values
if not filtered_ml.empty:
    filtered_ml['predicted_rating'] = rating_model.predict(X_new)
    top_pred = filtered_ml.sort_values(by='predicted_rating', ascending=False)[['name','predicted_rating','price']].head(5)
    st.sidebar.dataframe(top_pred)
else:
    st.sidebar.write("No games match the current filters.")

# KPI Cards ----------------
st.subheader("🎯 Results")
st.write(f"Showing {len(filtered_df)} games")
col1, col2, col3 = st.columns(3)
col1.markdown(f"<div class='kpi-card'>🎮<br>{len(filtered_df)}</div>", unsafe_allow_html=True)
col2.markdown(f"<div class='kpi-card'>⭐<br>{round(filtered_df['rating'].mean(),2)}</div>", unsafe_allow_html=True)
col3.markdown(f"<div class='kpi-card'>💰<br>{round(filtered_df['price'].mean(),2)}</div>", unsafe_allow_html=True)

# ---------------- Charts ----------------
st.subheader("📊 Insights")
col1, col2 = st.columns(2)

# Popular Games Scatter Plot
with col1:
    scatter_df = filtered_df[filtered_df['positive_ratings'] >= 50]
    fig = px.scatter(
        scatter_df,
        x='price',
        y='rating',
        size='positive_ratings',
        color='rating',
        color_continuous_scale=['#ff4d4d','#a64dff','#4da6ff'],
        hover_name='name',
        hover_data={'price':True, 'rating':True, 'positive_ratings':True},
        labels={'price':'Price ($)', 'rating':'Rating (0-5)', 'positive_ratings':'Positive Ratings'},
        title="💎 Popular Game Ratings vs Price"
    )
    fig.update_traces(marker=dict(line=dict(width=1,color='DarkSlateGrey'), opacity=0.7))
    fig.update_layout(xaxis=dict(title='Price ($)', gridcolor='LightGray'),
                      yaxis=dict(title='Rating (0-5)', gridcolor='LightGray'),
                      template='plotly_dark', transition_duration=500)
    st.plotly_chart(fig, use_container_width=True)

# Genre Distribution Bar Chart
with col2:
    genre_count = defaultdict(int)
    genre_rating_sum = defaultdict(float)
    for _, row in filtered_df.iterrows():
        if pd.isna(row['genres']): continue
        for g in row['genres'].split(';'):
            genre_count[g] += 1
            genre_rating_sum[g] += row['rating']
    avg_rating_per_genre = {g: genre_rating_sum[g]/genre_count[g] for g in genre_count}
    top_genres = sorted(genre_count.items(), key=lambda x: x[1], reverse=True)[:10]
    x = [g[0] for g in top_genres]
    y = [g[1] for g in top_genres]
    colors = [avg_rating_per_genre[g[0]] for g in top_genres]
    fig2 = px.bar(
        x=x, y=y, color=colors,
        color_continuous_scale=['#ff4d4d','#a64dff','#4da6ff'],
        labels={'x':'Genre','y':'Number of Games','color':'Avg Rating'},
        title="🎮 Top 10 Genres by Number of Games"
    )
    fig2.update_layout(transition_duration=500)
    st.plotly_chart(fig2, use_container_width=True)

# ---------------- Top Rated Games ----------------
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
st.subheader("🏆 Top Rated Games")
top_rated = filtered_df.sort_values(by='rating', ascending=False)[['name','rating','price']]
st.dataframe(top_rated.head(10))
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Hidden Gems ----------------
st.markdown("<div class='fade-in'>", unsafe_allow_html=True)
st.subheader("💎 Hidden Gems")
hidden = filtered_ml[(filtered_ml['rating'] < filtered_ml['predicted_rating']) & (filtered_ml['rating'] >= 2.5)]
st.dataframe(hidden.sort_values(by='predicted_rating', ascending=False)[['name','rating','predicted_rating','price']].head(10))
st.markdown("</div>", unsafe_allow_html=True)