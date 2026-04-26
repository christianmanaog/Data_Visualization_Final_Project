# ==============================================================================
# SECTION 1: DATA ACQUISITION AND STRUCTURAL VALIDATION
# ==============================================================================
# In this section, we ingest the raw Amazon product review dataset and 
# perform an initial structural assessment to identify missing values and 
# verify data types.
# ==============================================================================
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from IPython.display import display 

# Ingest the primary 100k dataset
print("Initializing data ingestion...")
df = pd.read_csv('amazon_reviews.csv')

# Execute structural validation (schema, non-null counts, and data types)
print("\n--- DataFrame Schema and Information ---")
df.info()

print("\n--- Initial Missing Value Counts ---")
print(df.isnull().sum())


# ==============================================================================
# SECTION 2: DATA PREPROCESSING AND FEATURE ENGINEERING
# ==============================================================================
# This phase focuses on data cleaning, handling null values, and performing 
# necessary type conversions. We also engineer preliminary features like 
# 'Sales_Volume_Proxy', 'Review_Word_Count', and 'Product_Label'.
# ==============================================================================

# 1. Null Value Mitigation
df_clean = df.dropna(subset=['ProfileName', 'Summary']).copy()

# 2. Temporal Data Conversion
df_clean['Review_Date'] = pd.to_datetime(df_clean['Time'], unit='s')
df_clean['Year_Month'] = df_clean['Review_Date'].dt.to_period('M').astype(str)

# 3. Target Variable Engineering (Sales Proxy)
sales_proxy = df_clean.groupby(['ProductId', 'Year_Month']).size().reset_index(name='Sales_Volume_Proxy')
df_clean = pd.merge(df_clean, sales_proxy, on=['ProductId', 'Year_Month'], how='left')

# 4. Text Meta-Feature Engineering
df_clean['Review_Word_Count'] = df_clean['Text'].astype(str).str.split().str.len()

# 5. Data Enrichment: Targeted Product Mapping (100k Dataset Only)
# We perfectly map our high-volume targets and outliers based strictly on the 100k dataset. 
# We DO NOT remove the original Product IDs. We just create a label for the graphs.
# 5. Data Enrichment: Targeted Product Mapping (100k Dataset Only)
# We perfectly map our high-volume targets and outliers based strictly on the 100k dataset. 
# We DO NOT remove the original Product IDs. We just create a label for the graphs.
product_mapping = {
    'B0026RQTGE': 'Greenies Dental Chews (36oz)',
    'B002QWP89S': 'Greenies Dental Chews (27oz)', 
    'B0013NUGDE': 'Popchips Potato Chips',
    'B007M83302': 'Popchips Variety Pack',
    'B000KV61FC': 'PetSafe Tug-A-Jug Dog Toy',
    'B005ZBZLT4': 'SF Bay Coffee K-Cups',
    'B002IEZJMA': 'Illy Ready-to-Drink Coffee',
    'B000PDY3P0': 'Snappy Popcorn Kernels',
    'B002LANN56': 'Chef Michaels Dry Dog Food',
    'B004SRH2B6': 'Zico Pure Coconut Water',
    'B006N3IG4K': 'Keurig K-Cup Coffee (Blend 1)',
    'B003VXFK44': 'Keurig K-Cup Coffee (Blend 2)',
    'B0041NYV8E': 'Gold Kili Ginger & Lemon Mix',
    'B0007A0AQW': 'Peanut Butter Dog Treats',
    'B0018KR8V0': 'LÄRABAR Energy/Meal Bar',
    'B001LG940E': 'Purina Pet Food (Promo)',
    'B006H34CUS': 'Quaker Banana Nut Bread Mix',
    'B001LGGH40': 'The Switch Orange Tangerine Soda',
    'B0045XE32E': 'Lamb & Barley Dog Treats',
    'B002NHYQAS': 'Premium Chocolate Assortment',
    'B007I7Z3Z0': 'Iced Black Tea with Lemon',
    'B004YV80O4': 'Macaroni & Cheese Dinner',
    'B001LG945O': 'The Switch Beverage (Multipack)',
    'B004MO6NI8': 'Energy Drink Assortment',
    'B001EQ55RW': 'Blue Diamond Cocoa Almonds',
    'B004ZIER34': 'Puroast Low Acid Coffee',
    'B001BDDTB2': 'Petite Cuisine Cat Food'
}

# Apply the perfect manual mapping to our new Graph Label column. 
# If it's not in our dictionary, the graph will just show the ID.
df_clean['Product_Label'] = df_clean['ProductId'].map(product_mapping).fillna("ID: " + df_clean['ProductId'])

# Apply the perfect manual mapping to our new Graph Label column. 
# If it's not in our Top 15 dictionary, the graph will just show the ID.
df_clean['Product_Label'] = df_clean['ProductId'].map(product_mapping).fillna("ID: " + df_clean['ProductId'])


# ==============================================================================
# SECTION 3: NATURAL LANGUAGE PROCESSING & SENTIMENT EXTRACTION
# ==============================================================================
# Utilizing the TextBlob lexicon-based approach, we process the unstructured 
# 'Text' column to derive numerical sentiment polarity scores ranging 
# from -1.0 (highly negative) to 1.0 (highly positive).
# ==============================================================================

def extract_sentiment_polarity(text):
    """
    Computes the sentiment polarity of a given text string.
    Returns a float representing polarity (-1.0 to 1.0).
    """
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0

print("\nExecuting sentiment polarity extraction... (Computation in progress)")
# Apply the sentiment extraction algorithm
df_clean['Sentiment_Polarity'] = df_clean['Text'].apply(extract_sentiment_polarity)

# Categorize sentiment into discrete buckets for classification analysis
df_clean['Sentiment_Class'] = pd.cut(df_clean['Sentiment_Polarity'], 
                                     bins=[-1.01, -0.1, 0.1, 1.01], 
                                     labels=['Negative', 'Neutral', 'Positive'])

print("\nSample of Processed Data with Sentiment Features:")
# Notice we keep both ProductId and Product_Label in the final dataset!
display(df_clean[['ProductId', 'Product_Label', 'Score', 'Review_Word_Count', 'Sentiment_Polarity']].head())


# ==============================================================================
# SECTION 4: PRELIMINARY EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
# Formulating 10 initial exploratory visualizations to identify distribution 
# anomalies, potential outliers, and preliminary correlations.
# ==============================================================================

# EDA 1: Global Distribution of Sentiment Polarity
fig1 = px.histogram(df_clean, x='Sentiment_Polarity', nbins=50, 
                    title='EDA 1: Global Distribution of Sentiment Polarity',
                    color_discrete_sequence=['#636EFA'])
fig1.show()

# EDA 2: Sentiment Polarity vs. Ordinal Star Rating
fig2 = px.box(df_clean, x='Score', y='Sentiment_Polarity', 
              title='EDA 2: Distribution of Sentiment Polarity by Star Rating',
              category_orders={"Score": [1, 2, 3, 4, 5]})
fig2.show()

# EDA 3: Macro-Level Sentiment Trends Over Time
monthly_sentiment = df_clean.groupby('Year_Month')['Sentiment_Polarity'].mean().reset_index()
fig3 = px.line(monthly_sentiment, x='Year_Month', y='Sentiment_Polarity', 
               title='EDA 3: Longitudinal Aggregation of Mean Sentiment Polarity')
fig3.update_xaxes(tickangle=45)
fig3.show()

# EDA 4: Correlation: Sentiment vs. Helpfulness
df_helpful = df_clean[df_clean['HelpfulnessDenominator'] >= 5].copy()
df_helpful['Helpfulness_Ratio'] = df_helpful['HelpfulnessNumerator'] / df_helpful['HelpfulnessDenominator']
fig4 = px.scatter(df_helpful, x='Sentiment_Polarity', y='Helpfulness_Ratio', opacity=0.3, 
                  title='EDA 4: Correlation: Sentiment vs. Helpfulness', trendline="ols")
fig4.show()

# EDA 5: Review Word Count vs. Star Rating
fig5 = px.box(df_clean, x='Score', y='Review_Word_Count', 
              title='EDA 5: Review Length (Word Count) Distribution by Star Rating',
              category_orders={"Score": [1, 2, 3, 4, 5]}, points=False)
fig5.update_yaxes(range=[0, 300]) 
fig5.show()

# EDA 6: Sentiment Class Proportion Over Time (Stacked Bar)
sent_class_time = df_clean.groupby(['Year_Month', 'Sentiment_Class'], observed=False).size().reset_index(name='Count')
fig6 = px.bar(sent_class_time, x='Year_Month', y='Count', color='Sentiment_Class', 
              title='EDA 6: Proportion of Sentiment Categories Over Time',
              color_discrete_map={'Negative':'red', 'Neutral':'gray', 'Positive':'green'})
fig6.update_xaxes(tickangle=45)
fig6.show()

# EDA 7: Global Review Volume Over Time (Sales Proxy Growth)
monthly_vol = df_clean.groupby('Year_Month').size().reset_index(name='Total_Reviews')
fig7 = px.area(monthly_vol, x='Year_Month', y='Total_Reviews', 
               title='EDA 7: Total Monthly Review Volume (Proxy for Sales Velocity)')
fig7.update_xaxes(tickangle=45)
fig7.show()

# EDA 8: Top 15 Best-Selling Products by Sentiment
# The graph uses Product_Label so the names look beautiful!
prod_agg = df_clean.groupby('Product_Label').agg(
    {'Sentiment_Polarity':'mean', 'Sales_Volume_Proxy':'max'}
).reset_index()

top_prods = prod_agg.nlargest(15, 'Sales_Volume_Proxy')
fig8 = px.bar(top_prods, x='Product_Label', y='Sales_Volume_Proxy', color='Sentiment_Polarity',
              title='EDA 8: Top 15 Highest Volume Products Colored by Average Sentiment',
              color_continuous_scale='RdYlGn')
fig8.update_xaxes(tickangle=45)
fig8.show()

# EDA 9: Product-Level Correlation (Sentiment vs Volume with Hover Info)
# When you hover over a dot, it shows either the real name or the ID!
fig9 = px.scatter(prod_agg, x='Sentiment_Polarity', y='Sales_Volume_Proxy', opacity=0.4,
                  title='EDA 9: Product-Level Correlation: Sentiment vs. Max Sales Volume Proxy',
                  hover_name='Product_Label',
                  trendline='ols')
fig9.show()

# EDA 10: 2D Density Heatmap of Sentiment vs Star Rating
fig10 = px.density_heatmap(df_clean, x='Score', y='Sentiment_Polarity', nbinsy=20,
                           title='EDA 10: 2D Density Heatmap of Sentiment Polarity vs. Ratings',
                           color_continuous_scale='Viridis')
fig10.show()


# ==============================================================================
# SECTION 5: DATA SERIALIZATION AND EXPORT
# ==============================================================================
# Committing the structured, cleaned, and feature-engineered DataFrame to 
# a standard CSV format to support the subsequent high-fidelity dashboard 
# development phase.
# ==============================================================================

# Isolate necessary features and omit redundant data to optimize storage
df_final = df_clean.drop(columns=['Time'])

# Serialize the structured dataframe to CSV
file_name = 'cleaned_amazon_reviews_with_sentiment.csv'
df_final.to_csv(file_name, index=False)
print(f"\nData pipeline execution complete. Serialized output saved to: {file_name}")