import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import itertools
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import os


def load_data():
    df = pd.read_csv("clean_listings.csv")
    return df

@st.cache_data
def load_embeddings():
    """
    loads and caches the embeddings for the semantic search
    """
    return np.load('description_embeddings_word2vec.npy')

@st.cache_data
def load_preprocessed_data():
    """
    loads and caches the data for the dashboard
    """
    df = pd.read_csv('preprocessed_df.csv')
    return df

def load_word2vec_model():
    return Word2Vec.load("word2vec_model.bin")

# Function to generate sentence embeddings by averaging word vectors
def sentence_embedding(sentence, word2vec_model):
    words = sentence.lower().split()
    vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
    if not vectors:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(vectors, axis=0)

def semantic_search(queries, corpus, corpus_embeddings, preprocessed_df, word2vec_model):
    # Perform semantic search
    top_k = min(5, len(corpus))
    results = []
    for query in queries:
        query_embedding = sentence_embedding(query, word2vec_model)

        # Calculate cosine similarity
        cos_scores = cosine_similarity([query_embedding], corpus_embeddings)[0]
        top_results_indices = np.argsort(cos_scores)[-top_k:][::-1]

        print("\n\n======================\n\n")
        print("Query:", query)
        print("\nTop 5 most similar sentences in corpus:")
        print(top_results_indices)

        for idx in top_results_indices:
            listing_id = preprocessed_df[preprocessed_df['original_description']==corpus[idx]]['id'].iloc[0]
            listing_url = preprocessed_df[preprocessed_df['id']==listing_id]['listing_url'].iloc[0]
            results.append((corpus[idx], listing_url, listing_id, cos_scores[idx]))
        return results

df = load_data()

st.title("Airbnb rental market in Copenhagen")

st.markdown(
    """
           
"""
)

with st.expander("📊 **Objective**"):
   st.markdown(       """
The purpose of this app is to give an overview and insights of the AirBnb market in Copenhagen."""
   )

with st.expander("📚How to Use the Dashboard "):
   st.markdown(
       """ 
    1. **Instructions to use the application** - On the sidebar there are filters where you can select your preferences.The results will be shown accordingly.
    2. **Visualize Data** - From the dropdown, select a visualization type to view patterns.
    3. **Insights & Recommendations** - Scroll down to see insights derived from the visualizations and actionable recommendations.
   """
    )

visualization_option = st.selectbox(
    "Select Visualization",
    [
        "Distribution of property types",
        "Comparison between price and different important features",
    ],
)

if visualization_option == "Distribution of property types":
    property_type_counts = df["property_type"].value_counts()
    filtered_property_types = property_type_counts[property_type_counts > 30]
    filtered_df = df[df["property_type"].isin(filtered_property_types.index)]
    filtered_property_type_counts = filtered_df["property_type"].value_counts()

    fig = px.bar(
        x=filtered_property_type_counts.index,
        y=filtered_property_type_counts,
        labels={"x": "Property Type", "y": "Number of apartments"},
        title="Distribution of Property Types",
        height=500,
    )
    fig.update_layout(xaxis=dict(tickangle=-45, tickmode="array"), showlegend=False)
    st.plotly_chart(fig)

if visualization_option == "Comparison between price and different important features":
    fig = plt.figure(figsize=(10, 6))
    sns.lineplot(x="bedrooms", y="price", data=df, label="Bedrooms", ci=False)
    sns.lineplot(x="beds", y="price", data=df, label="Beds", ci=False)
    sns.lineplot(x="bathrooms", y="price", data=df, label="Bathrooms", ci=False)
    sns.lineplot(x="accommodates", y="price", data=df, label="Accommodates", ci=False)
    plt.xlabel("Number of Bedrooms, Beds, Bathrooms, or Accommodates")
    plt.ylabel("Price")
    plt.title("Price vs Number of Bedrooms, Beds, Bathrooms, or Accommodates")
    plt.grid(True)
    plt.legend()
    st.pyplot(fig)

# NLP PART - LOADING THE DATA AND PREPROCESSING, TOPIC MODELLING VISUALISATION

# # Filters on the sidebar
selected_neighbourhood = df["neighbourhood_cleansed"].unique().tolist()
selected_neighbourhood = st.sidebar.multiselect(
    "Select a neighbourhood", selected_neighbourhood, default=[]
)
if not selected_neighbourhood:
    st.warning("Please select a neighbourhood from the sidebar")
    st.stop()
neighbourhood_filtered_df = df[
    df["neighbourhood_cleansed"].isin(selected_neighbourhood)
]

min_number_of_people = neighbourhood_filtered_df["accommodates"].min()
max_number_of_people = neighbourhood_filtered_df["accommodates"].max()
accommodation_range = st.sidebar.slider(
    "Select the number of people who are travelling",
    int(min_number_of_people),
    int(max_number_of_people),
    (1, 1),
)
accommodation_filtered_df = neighbourhood_filtered_df[
    (neighbourhood_filtered_df["accommodates"] >= accommodation_range[0])
    & (neighbourhood_filtered_df["accommodates"] <= accommodation_range[1])
]

# Interactive map of Copenhagen

st.title("Interactive map of Copenhagen")


gdf = gpd.GeoDataFrame(
    accommodation_filtered_df,
    geometry=gpd.points_from_xy(
        accommodation_filtered_df["longitude"], accommodation_filtered_df["latitude"]
    ),
)
gdf.crs = "EPSG:4326"
gdf = gdf.dropna(subset=["latitude", "longitude"])
copenhagen_coordinates = [55.6761, 12.5683]
m = folium.Map(location=copenhagen_coordinates, zoom_start=11, tiles="OpenStreetMap", width=800, height=600)
marker_cluster = MarkerCluster().add_to(m)

for _, row in gdf.iterrows():
    popup_content = f"""
    Nightly price: {row['nightly_price']}<br>
    Review rating: {row['review_scores_rating']}<br>
    Link to the apartment: {row['listing_url']}<br>
    """
    popup = folium.Popup(popup_content, max_width=300)

    folium.Circle(
        location=[row["latitude"], row["longitude"]],
        radius=5,
        color="blue",
        fill=True,
        fill_color="blue",
        fill_opacity=0.4,
        popup=popup,
    ).add_to(marker_cluster)

st_folium(m, width=800, height=600)


# Price prediction interface

def preprocess_features(df):
    X_1 = df.loc[:, ['accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights', 'availability_30', 'availability_365', 'host_listings_count']]
    features_to_encode = df[['neighbourhood_cleansed', 'property_type']].values
    onehot_encoder = OneHotEncoder(sparse=False)
    encoded_features = onehot_encoder.fit_transform(features_to_encode)
    columns_ohe = list(itertools.chain(*onehot_encoder.categories_))
    df_onehot = pd.DataFrame(encoded_features, columns=columns_ohe)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X_1.iloc[:, :])
    X_1.iloc[:,:] = scaled_features
    X_1.index = range(len(X_1))
    df_onehot.index = range(len(df_onehot))
    X = X_1.join(df_onehot)
    Y = df['nightly_price']
    return X, Y

def run_price_prediction(df):
    # Create a sidebar for user input
    st.sidebar.header('Airbnb Price Prediction')

    # Add filters or input options
    selected_bedrooms = st.sidebar.selectbox('Select the number of bedrooms', [1, 2, 3, 4, 5], index=0)  # Default to 3 bedrooms
    selected_bathrooms = st.sidebar.selectbox('Select the number of bathrooms', [1, 2, 3, 4, 5], index=0)  # Default to 2 bathrooms
    selected_accommodates = st.sidebar.selectbox('Select the number of people who will be accommodated', [1, 2, 3, 4, 5], index=0)  # Default to 3 people

    # For property type, you can use selectbox to choose only one type
    property_types = df['property_type'].unique()
    selected_property_type = st.sidebar.selectbox('Select property type', property_types, index=0)  # Default to the first property type

# Combine the selected features to filter the dataset
    filtered_df = df[(df['bedrooms'] == selected_bedrooms) &
                 (df['bathrooms'] == selected_bathrooms) &
                 (df['accommodates'] == selected_accommodates) &
                 (df['property_type'] == selected_property_type)]


    # Check if filtered_df is not empty and contains the necessary columns
    if not filtered_df.empty and 'nightly_price' in filtered_df.columns:
        X, Y = preprocess_features(filtered_df)
        model_xgb = XGBRegressor(booster='gbtree')
        y_pred = cross_val_predict(model_xgb, X, Y, cv=2, method='predict')

        # Display the predicted price
        predicted_price = round(y_pred.mean(), 2)
        st.subheader("", divider='rainbow')
        st.subheader("Predicted price based on your inputs: **€{:.2f}**".format(predicted_price))
        st.subheader("", divider='rainbow')
        #st.markdown()
    else:
        st.warning("No data found or missing 'nightly_price' column in the filtered dataset.")

run_price_prediction(df)

preprocessed_df = load_preprocessed_data()
corpus = preprocessed_df['original_description'].tolist()
corpus_embeddings = load_embeddings()
word2vec_model = load_word2vec_model()

plots_dir = "wordcloud_plots"
image_files = os.listdir(plots_dir)

st.title("Accomodation recommender")
#st.subheader("The workclouds represent three main factors: Accomodation Features,Location and Amenities")

image_titles = ["Accomodation Features","Amenities","Location"]



with st.expander("📊 **Purpose**"):
   st.markdown(       """
The wordclouds provide a visual representation of keywords associated with listings. Use them in order to get the most accurate results. They represent three main factors: Accomodation Features, Location and Amenities"""
   )

for img_file, image_title in zip(image_files, image_titles):
    img_path = os.path.join(plots_dir, img_file)
    st.header(image_title, divider='rainbow')
    st.image(img_path)

st.subheader("Please describe your accomodation preference:")
user_input = st.text_area(label= '', placeholder= "Your description")

col1, col2 = st.columns([1, 7])

# Submit button triggers the search
if col1.button("Submit"):
    results = semantic_search([user_input], corpus, corpus_embeddings, preprocessed_df, word2vec_model)
    for idx , (matched_description, listing_url, listing_id, score) in enumerate(results):
        confidence_percent = "{:.2%}".format(score)
        st.subheader(f'#{idx+1} likelihood to your preference: {confidence_percent}', divider='rainbow')
        st.components.v1.html(f'<div class="airbnb-embed-frame" data-id="{listing_id}" data-view="home" data-hide-price="true" style="width: 450px; height: 300px; margin: auto;"><a href="https://www.airbnb.com/rooms/44132738?guests=1&amp;adults=1&amp;s=66&amp;source=embed_widget">View On Airbnb</a><a href="https://www.airbnb.com/rooms/44132738?guests=1&amp;adults=1&amp;s=66&amp;source=embed_widget" rel="nofollow">Loft in Copenhagen · 1 bedroom · 1 bed · 1 bath</a><script async="" src="https://www.airbnb.com/embeddable/airbnb_jssdk"></script></div>', width=460, height=495, scrolling=False)
        st.markdown(matched_description)

if col2.button("Clear"):
        user_input=''