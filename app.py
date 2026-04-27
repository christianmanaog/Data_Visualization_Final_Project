# ==============================================================================
# SECTION 1: DATA ACQUISITION AND STRUCTURAL VALIDATION
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
# Includes both the Top Volume Sellers AND the Bottom Sentiment Performers
product_mapping = {
    # Original Top Sellers
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
    'B001BDDTB2': 'Petite Cuisine Cat Food',
    
    # NEW: The 15 Lowest Sentiment Products (Min 10 reviews)
    'B004ET5TP4': 'Peppermint Mocha Mix',
    'B001E96JY2': 'Quick Lunch Ready Meal',
    'B003SBRUC4': 'Gourmet Food Item',
    'B000UGXWQ8': 'Tutti Frutti Sticks (80ct)',
    'B000UH3QWW': 'Tutti Frutti Sticks',
    'B000I5FN7C': 'Crunchy Dog Treats',
    'B000YPKODY': 'Cracker Jack Snacks',
    'B004EI4C7G': 'Weight Loss Supplement',
    'B004G5ZYN8': 'Dog Chew Toy (Var A)',
    'B004G5ZYQA': 'Dog Chew Toy (Var B)',
    'B004G5ZYQU': 'Dog Chew Toy (Var C)',
    'B0048HWXA6': 'Gag Gift Jelly Beans',
    'B002TSA90C': 'Generic Hardware/Item',
    'B001DW2RGO': '6-Hour Power Energy Drink',
    'B000FPGZFY': 'Baby Food Purée'
}

# Apply the perfect manual mapping to our new Graph Label column. 
# If it's not in our dictionary, the graph will just show the ID.
df_clean['Product_Label'] = df_clean['ProductId'].map(product_mapping).fillna("ID: " + df_clean['ProductId'])


# ==============================================================================
# SECTION 3: NATURAL LANGUAGE PROCESSING & SENTIMENT EXTRACTION
# ==============================================================================
def extract_sentiment_polarity(text):
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
display(df_clean[['ProductId', 'Product_Label', 'Score', 'Review_Word_Count', 'Sentiment_Polarity']].head())


# ==============================================================================
# SECTION 4: PRELIMINARY EXPLORATORY DATA ANALYSIS (EDA)
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

# NEW -> EDA 11: Bottom 15 Products by Lowest Average Sentiment
# We calculate total reviews to filter out products with only 1 or 2 bad reviews.
prod_agg_bottom = df_clean.groupby('Product_Label').agg(
    {'Sentiment_Polarity': 'mean', 'ProductId': 'count'}
).rename(columns={'ProductId': 'Total_Reviews'}).reset_index()

# Filter for at least 10 reviews, then grab the 15 lowest scores
bottom_prods = prod_agg_bottom[prod_agg_bottom['Total_Reviews'] >= 10].nsmallest(15, 'Sentiment_Polarity')

fig11 = px.bar(bottom_prods, x='Product_Label', y='Sentiment_Polarity', color='Sentiment_Polarity',
               title='EDA 11: Top 15 Products with the Lowest Average Sentiment (Min 10 Reviews)',
               color_continuous_scale='Reds_r') # We use red to visually highlight negative sentiment
fig11.update_xaxes(tickangle=45)
fig11.show()


# ==============================================================================
# SECTION 5: DATA SERIALIZATION AND EXPORT
# ==============================================================================

# Isolate necessary features and omit redundant data to optimize storage
df_final = df_clean.drop(columns=['Time'])

# Serialize the structured dataframe to CSV
file_name = 'cleaned_amazon_reviews_with_sentiment.csv'
df_final.to_csv(file_name, index=False)
print(f"\nData pipeline execution complete. Serialized output saved to: {file_name}")