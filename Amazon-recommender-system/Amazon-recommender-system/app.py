from typing import List, Tuple
import streamlit as st
import pickle
import pandas as pd
import mysql.connector

# Load precomputed matrices and rules from pickle files
cosinesimilarity_matrix = pickle.load(open('cosinesimilarity_matrix.pkl', 'rb'))
item_similarity_df = pickle.load(open('item_similarity_df.pkl', 'rb'))
user_item_matrix = pickle.load(open('user_item_matrix.pkl', 'rb'))
rules = pickle.load(open('rules.pkl', 'rb'))
monthly_support = pickle.load(open('monthly_support.pkl', 'rb'))

# Function to connect to the MySQL database
def connect_to_db():
    conn = mysql.connector.connect(
        # Using MySQL connector to establish a connection to the local database
        host="localhost",
        user="root",
        password="Irwa@2024",
        database="test"
    )
    return conn

# Function to load product data from the database
@st.cache_data(ttl=600)
def load_product_data() -> pd.DataFrame:
    conn = connect_to_db()
    query = "SELECT ProdID, ProdName, ImageURL FROM products"
    product_df = pd.read_sql(query, conn)
    conn.close()
    return product_df

# Function to load user data and their transactions from the database
@st.cache_data(ttl=600)
def load_user_data() -> pd.DataFrame:
    conn = connect_to_db()
    query = "SELECT Username, Transactions FROM transactions"
    user_data = pd.read_sql(query, conn)
    conn.close()
    user_data['Transactions'] = user_data['Transactions'].apply(lambda x: x.split(',') if isinstance(x, str) else [])
    return user_data

# Load product and user data from the database
product_df = load_product_data()
user_df = load_user_data()

### Function to recommend similar products using content-based filtering
def similar_products(product_name: str, top_n: int = 5) -> List[Tuple[str, str]]:
    try:
        # Find the index of the product with the matching name in the dataframe
        product_index = product_df[product_df['ProdName'].str.contains(product_name, case=False, na=False)].index[0]
    except IndexError:
        # Handle cases where no product is found
        return []
    # Get similarity scores for the product using the cosine similarity matrix
    distances = cosinesimilarity_matrix[product_index]
    # Sort and select top_n most similar products
    products_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n + 1]
    # Fetch recommended product names and images
    recommended_products = product_df.iloc[[i[0] for i in products_list]][['ProdID', 'ProdName', 'ImageURL']]
    #  Extract the first image URL for each product
    for index, row in recommended_products.iterrows():
        image_urls = row['ImageURL']
        image_urls_list = image_urls.split(',') if isinstance(image_urls, str) else [image_urls]
        first_image_url = image_urls_list[0].strip()
        recommended_products.at[index, 'ImageURL'] = first_image_url

    return [(row['ProdName'], row['ImageURL']) for _, row in recommended_products.iterrows()]


### Function to recommend items using item-item collaborative filtering
def recommend_items_item_based(user: str, top_n: int = 4) -> List[Tuple[str, str]]:
    # Check if user exists in the user-item matrix
    if user not in user_item_matrix.index:
        popular_items = user_item_matrix.sum().sort_values(ascending=False).head(top_n)

        recommendations = []
        for item in popular_items.index:
            # Fetch product name and image for each popular item
            prod_name = product_df.loc[product_df['ProdID'] == item, 'ProdName'].values[0]
            image_url = product_df.loc[product_df['ProdID'] == item, 'ImageURL'].values[0]

            image_urls_list = image_url.split(',') if isinstance(image_url, str) else [image_url]
            first_image_url = image_urls_list[0].strip()

            recommendations.append((prod_name, first_image_url))

        return recommendations

    # Get user's ratings and compute similarity scores with other items
    user_ratings = user_item_matrix.loc[user]
    similar_items = item_similarity_df.dot(user_ratings).sort_values(ascending=False)
    # Recommend items that the user has not already rated
    recommended_items = similar_items[~user_ratings.index.isin(user_ratings[user_ratings > 0].index)].head(top_n)

    recommendations = []
    for item in recommended_items.index:
        # Fetch product name and image for each recommended item
        prod_name = product_df.loc[product_df['ProdID'] == item, 'ProdName'].values[0]
        image_url = product_df.loc[product_df['ProdID'] == item, 'ImageURL'].values[0]

        image_urls_list = image_url.split(',') if isinstance(image_url, str) else [image_url]
        first_image_url = image_urls_list[0].strip()

        recommendations.append((prod_name, first_image_url))

    return recommendations

### Function to recommend products based on association rules (user's previous purchases)
def recommend_products_by_user(username: str, top_n: int = 5) -> List[Tuple[str, str]]:
    user_data = user_df[user_df['Username'] == username]

    if user_data.empty:
        return [("No data available for this user.", "No Image URL")]

    # Fetch the user's transactions
    transactions = user_data['Transactions'].values[0]

    # Convert transactions to a list format if necessary
    if isinstance(transactions, str):
        user_purchases = transactions.split(',')
    elif isinstance(transactions, list):
        user_purchases = transactions  # If it's already a list
    else:
        return [("Invalid transactions format.", "No Image URL")]

    # Remove duplicates
    user_purchases_set = set(user_purchases)

    recommendations = set()

    # Loop through association rules and find recommendations
    for _, rule in rules.iterrows():
        antecedent = rule['antecedents']

        if isinstance(antecedent, frozenset):
            antecedent = {item for item in antecedent}
            intersection = antecedent.intersection(user_purchases_set)

            if intersection:
                consequents = rule['consequents']
                if isinstance(consequents, frozenset):
                    recommendations.update(consequents)
                elif isinstance(consequents, str):
                    recommendations.update([item for item in consequents.split(',')])

    # Remove already purchased items from recommendations
    recommendations = list(recommendations - user_purchases_set)

    if not recommendations:
        return [("No recommendations found.", "No Image URL")]

    recommended_products_with_images = []
    # Get the product info (name, image URL) for each recommendation
    for rec in recommendations[:top_n]:
        product_info = product_df[product_df['ProdName'] == rec]

        if not product_info.empty:
            image_urls = product_info['ImageURL'].values[0]
            image_urls_list = image_urls.split(',') if isinstance(image_urls, str) else [image_urls]
            first_image_url = image_urls_list[0].strip()

            recommended_products_with_images.append((rec, first_image_url))
        else:
            recommended_products_with_images.append((rec, "No Image URL"))

    return recommended_products_with_images


### Function to recommend items based on the selected month
def recommend_items_month(selected_month: str, top_n: int = 5) -> List[Tuple[str, str]]:
    # Convert the selected month string to a datetime object for manipulation
    selected_month_dt = pd.to_datetime(selected_month)

    # Initialize previous_year_month as one year before the selected month
    previous_year_month = (selected_month_dt - pd.DateOffset(years=1)).strftime('%Y-%m')

    # Loop to find a valid previous year month in the monthly_support DataFrame
    while previous_year_month not in monthly_support.index:
        selected_month_dt -= pd.DateOffset(years=1)
        previous_year_month = selected_month_dt.strftime('%Y-%m')

        # Exit if the year has dropped below a certain limit to avoid infinite loops
        if selected_month_dt.year < 2000:
            return []  # Exit the function if no valid month is found

    # Extract the support values for the found previous year's selected month
    answer = monthly_support.loc[previous_year_month]

    # Rank the items by their support values in descending order
    ranked_items = answer.sort_values(ascending=False)

    # Get the top N product IDs and their corresponding names and images
    top_items = ranked_items.head(top_n).index.tolist()
    recommendations = []

    for item_name in top_items:
        product_info = product_df[product_df['ProdName'] == item_name]

        if not product_info.empty:
            prod_name = product_info['ProdName'].values[0]
            image_urls = product_info['ImageURL'].values[0]
            first_image_url = image_urls.split(',')[0].strip() if isinstance(image_urls, str) else image_urls
            recommendations.append((prod_name, first_image_url))

    return recommendations


### Hybrid recommendation system combining user-based, item-based, and content-based recommendations
def hybrid_recommendation(user, product_name, w_user_user=0.4, w_item_item=0.3, w_content=0.2):
    # Fetch recommendations from each recommendation type
    user_recommendations = recommend_products_by_user(user)  # User-based
    item_item_score = recommend_items_item_based(user)  # Item-based
    content_score = similar_products(product_name)  # Content-based

    # Handle potential None return from similar_products and seasonal_recommendations
    if content_score is None:
        content_score = []

    # Create sets of product names from each recommendation list
    user_user_set = set([prod for prod, _ in user_recommendations])
    item_item_set = set([prod for prod, _ in item_item_score])
    content_set = set([prod for prod, _ in content_score])

    # Weighted combination of all recommendations
    final_recommendations = {}
    for item in user_user_set.union(item_item_set).union(content_set):
        score = 0
        if item in user_user_set:
            score += w_user_user
        if item in item_item_set:
            score += w_item_item
        if item in content_set:
            score += w_content

        final_recommendations[item] = score

    # Retrieve product names and image URLs for the top recommendations
    sorted_recommendations = sorted(final_recommendations.items(), key=lambda x: x[1], reverse=True)

    final_result = []
    for item, score in sorted_recommendations:
        # Get image URL from product_df
        product_info = product_df[product_df['ProdName'] == item]
        if not product_info.empty:
            image_url = product_info['ImageURL'].values[0].split(',')[0].strip()
            final_result.append((item, image_url))

    return final_result


### Streamlit UI components to interact with the recommendation system
st.title('Product Recommendation System')

# Input to get the username
col1, col2 = st.columns(2)
with col1:
    text_input = st.text_input("Enter Username:")

# Dropdown to select a product
selected_product_name = st.selectbox(
    "Select a product to get recommendations",
    product_df['ProdName']
)
# Button to get hybrid recommendations based on user input
if st.button("Recommend"):

    # Get hybrid recommendations
    hybrid_recommendations = hybrid_recommendation(text_input, selected_product_name)

    if hybrid_recommendations:
        st.write("### Recommended Products")

        # Create a DataFrame for display
        display_df = pd.DataFrame({
            "Image": [f'<img src="{img}" width="150" />' for _, img in hybrid_recommendations],
            "Product Name": [name for name, _ in hybrid_recommendations]
        })

        # Display the DataFrame as HTML with images
        st.markdown(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
    else:
        st.write(f"No recommendations found for '{selected_product_name}'.")


current_date = pd.to_datetime("now")
selected_month = current_date.strftime('%Y-%m')
monthly_recommendations = recommend_items_month(selected_month)

if monthly_recommendations:
    st.write("### Monthly Recommended Products")

    # Create a DataFrame for display with images and product names
    monthly_display_df = pd.DataFrame({
        "Image": [f'<img src="{img}" width="150" />' for _, img in monthly_recommendations],
        "Product Name": [name for name, _ in monthly_recommendations]
    })

    # Display the DataFrame as HTML with images
    st.markdown(monthly_display_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.write("No monthly recommendations found.")



