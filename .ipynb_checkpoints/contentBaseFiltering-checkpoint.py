#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[96]:


dataset = pd.read_csv('Amazon dataset.csv')


# In[97]:


dataset.head()


# In[100]:


dataset = dataset[['id','name','categories','imageURLs','reviews.rating','reviews.username','reviews.date']]


# In[102]:


dataset.head()


# In[104]:


dataset.columns


# In[106]:


# Define the mapping of current column names to shorter names
column_name_mapping = {
'id': 'ProdID',
'name': 'ProdName',
'categories': 'Category',
'imageURLs': 'ImageURL',
'reviews.rating': 'Rating',
'reviews.username': 'Username',
'Product Image Url': 'ImageURL',
    'reviews.date' : 'Date',
}
# Rename the columns using the mapping
dataset.rename(columns=column_name_mapping, inplace=True)


# In[108]:


dataset.head()


# In[110]:


product = dataset.groupby(['ProdID','ProdName','Category', 'ImageURL' ]) ['Rating'].mean().reset_index()
product['Rating'] = product['Rating'].astype(int)


# In[112]:


product.head()


# In[114]:


product.shape


# In[116]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
 
# Download required resources
nltk.download('punkt')
nltk.download('stopwords')
 
# Define stop words
stop_words = set(stopwords.words('english'))
 
def process_text(text):
    text = text.casefold()  # Lowercase the text using casefold (more aggressive than lower())
    tokens = word_tokenize(text)  # Tokenize the text into words
    cleaned_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]  # Remove stop words
    return cleaned_tokens
 
# Apply the function to the 'ProdName' column
product['ProdName'] = product['ProdName'].apply(lambda x: process_text(x))
product['Category'] = product['Category'].apply(lambda x: process_text(x))


# In[117]:


product.head()


# In[120]:


# Join list elements into strings for 'ProdName' and 'Category'
product['ProdName'] = product['ProdName'].apply(lambda x: ' '.join(x))  # Joining list elements in ProdName
product['Category'] = product['Category'].apply(lambda x: ' '.join(x))  # Joining list elements in Category
 
# Convert necessary columns to strings
product['ProdID'] = product['ProdID'].astype(str)
product['Rating'] = product['Rating'].astype(str)
 
# Create the 'Tags' column by concatenating the relevant columns
product['Tags'] = product['ProdID'] + ' ' + product['ProdName'] + ' ' + product['Category'] + ' ' + product['Rating']


# In[122]:


product.head()


# In[130]:


product.head()


# In[132]:


product.shape


# In[159]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

# Transform the 'Tags' column into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(product['Tags']).toarray()
tfidf_matrix


# In[163]:


from sklearn.metrics.pairwise import cosine_similarity

# Compute cosine similarity matrix
cosinesimilarity_matrix = cosine_similarity(tfidf_matrix)
cosinesimilarity_matrix


# In[171]:


# Function to recommend similar products

def similar_products(product_name, product_df, cosinesimilarity_matrix, top_n=5):
    # Check if the product exists in the dataframe
    try:
        product_index = product_df[product_df['ProdName'].str.contains(product_name, case=False, na=False)].index[0]
    except IndexError:
        print(f"No product found matching '{product_name}' in the dataset.")
        return

    # Get cosine similarity scores for the selected product
    distances = cosinesimilarity_matrix[product_index]

    # Get most similar products excluding the input product itself
    products_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:top_n+1]

    # Print the top recommended products
    print(f"\nProducts similar to '{product_name}':")
    for i in products_list:
        print(product_df.iloc[i[0]]['ProdName'])

# Transform 'Tags' into a TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(product['Tags']).toarray()

# Compute cosine similarity matrix
cosinesimilarity_matrix = cosine_similarity(tfidf_matrix)


# In[173]:


# Example
similar_products('Laptop', product, cosine_sim_matrix)


# In[ ]:





# In[ ]:





# In[ ]:




