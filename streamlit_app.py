import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your cleaned dataset
df = pd.read_csv("netflix_cleaned.csv") 
st.title("🎬 Netflix Content Explorer")

# Sidebar filters
type_filter = st.sidebar.multiselect("Select Type:", df['type_SHOW'].unique(), default=df['type_SHOW'].unique())
year_range = st.sidebar.slider("Release Year:", int(df['release_year'].min()), int(df['release_year'].max()), (2010, 2023))
score_range = st.sidebar.slider("IMDb Score Range:", 0.0, 10.0, (6.0, 9.0))

# Filter data
filtered_df = df[
    (df['type_SHOW'].isin(type_filter)) &
    (df['release_year'].between(year_range[0], year_range[1])) &
    (df['imdb_score'].between(score_range[0], score_range[1]))
]

st.subheader("Filtered Netflix Titles")
st.write(f"Total Results: {filtered_df.shape[0]}")
st.dataframe(filtered_df)

# Optional plot
st.subheader("Rating Distribution")
fig, ax = plt.subplots()
sns.histplot(filtered_df['imdb_score'], bins=20, ax=ax, color='red')
st.pyplot(fig)
# High vs Low Rated Pie Chart
st.subheader("🎯 High vs Low Rated Content")

rating_counts = df['is_high_rated'].value_counts()
labels = ['High Rated (≥ 7)', 'Low Rated (< 7)']
fig3, ax3 = plt.subplots()
ax3.pie(rating_counts, labels=labels, autopct='%1.1f%%', colors=['green', 'red'], startangle=90)
ax3.axis('equal')
st.pyplot(fig3)

import joblib
model = joblib.load("imdb_regression_model.pkl")

st.sidebar.header("🎯 Predict IMDb Score")

input_year = st.sidebar.slider("Release Year", 1980, 2025, 2020)
input_runtime = st.sidebar.slider("Runtime (minutes)", 30, 240, 90)
input_type = st.sidebar.selectbox("Type", ["MOVIE", "SHOW"])

# Convert type to numeric (same encoding as training)
type_show = 1 if input_type == "SHOW" else 0

if st.sidebar.button("Predict Score"):
    input_df = pd.DataFrame([[input_year, input_runtime, type_show]], columns=['release_year', 'runtime', 'type_SHOW'])
    predicted_score = model.predict(input_df)[0]
    st.sidebar.success(f"🎬 Predicted IMDb Score: {predicted_score:.2f}")
