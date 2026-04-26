import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Amazon Reviews Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# ANALOGOUS COLOR PALETTE  (teal → blue-teal → amber)
# Positive:  Teal              #1D9E75
# Neutral:   Blue-teal         #1A7FA0
# Negative:  Muted amber       #D48B2A  (replaces red — stays analogous)
# Sales:     Deep teal         #0F6E56
# Trend:     Steel blue-green  #1A7FA0
# ==============================================================================
COLORS = {
    "positive":  "#1D9E75",
    "neutral":   "#1A7FA0",
    "negative":  "#D48B2A",
    "sales":     "#0F6E56",
    "trend":     "#1A7FA0",
    "teal_seq":  ["#E1F5EE", "#9FE1CB", "#5DCAA5", "#1D9E75", "#0F6E56", "#085041"],
}

FONT = "Inter, system-ui, sans-serif"


# ==============================================================================
# DATA LOADING
# ==============================================================================
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_amazon_reviews_with_sentiment.csv")
    df["Review_Date"] = pd.to_datetime(df["Review_Date"])
    df["Year"] = df["Review_Date"].dt.year
    df["Sentiment_Class"] = df["Sentiment_Class"].astype(str)

    # Prefer a human-readable product name when the dataset provides one.
    # If absent, derive a stakeholder-friendly name from product review summaries.
    product_name_candidates = ["ProductName", "Product_Name", "ProductTitle", "Product_Title", "Title", "Name"]
    product_name_col = next((col for col in product_name_candidates if col in df.columns), None)
    if product_name_col:
        df["Product_Display"] = (
            df[product_name_col]
            .astype(str)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
            .str.replace(r"[^A-Za-z0-9&'(),./\-\s]", "", regex=True)
            .str.title()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        )
    else:
        df["Product_Display"] = pd.NA

    if "Summary" in df.columns:
        summary_name_map = (
            df.assign(Summary=df["Summary"].astype(str).str.strip())
            .loc[lambda x: x["Summary"].ne("") & x["Summary"].ne("nan")]
            .groupby("ProductId")["Summary"]
            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
            .str.replace(r"\s+", " ", regex=True)
            .str.replace(r"[^A-Za-z0-9&'(),./\-\s]", "", regex=True)
            .str.title()
            .str.slice(0, 48)
        )
        df["Product_Display"] = df["Product_Display"].fillna(df["ProductId"].map(summary_name_map))

    df["Product_Display"] = df["Product_Display"].fillna("Unknown Product")
    return df


df = load_data()


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown("### Filters")
    years = sorted(df["Year"].unique())
    selected_years = st.multiselect("Year range", years, default=years)
    selected_ratings = st.multiselect("Star ratings", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])
    selected_sentiments = st.multiselect(
        "Sentiment class", ["Positive", "Neutral", "Negative"],
        default=["Positive", "Neutral", "Negative"]
    )
    st.markdown("---")
    st.caption("Amazon Fine Food Reviews\n99,990 reviews · TextBlob sentiment")

# Apply filters
dff = df[
    df["Year"].isin(selected_years) &
    df["Score"].isin(selected_ratings) &
    df["Sentiment_Class"].isin(selected_sentiments)
].copy()

# ==============================================================================
# STYLE HELPER
# ==============================================================================
def style(fig, height=320, mb=40, ml=50):
    fig.update_layout(
        height=height,
        template="plotly_white",
        font=dict(family=FONT, size=11),
        margin=dict(l=ml, r=16, t=mb, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#EBEBEB", linecolor="#D0D0D0", tickfont=dict(size=10))
    fig.update_yaxes(showgrid=True, gridcolor="#EBEBEB", linecolor="#D0D0D0", tickfont=dict(size=10))
    return fig


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown("## Consumer Sentiment & E-commerce Lead Indicator")
st.caption("Analyze 99,990 Amazon reviews to visualize the relationship between sentiment polarity and sales-volume proxy across products over time.")
st.markdown("---")


# ==============================================================================
# KPI ROW — Styled cards
# ==============================================================================
total     = len(dff)
pct_pos   = (dff["Sentiment_Class"] == "Positive").mean() * 100
pct_neg   = (dff["Sentiment_Class"] == "Negative").mean() * 100
pct_neu   = (dff["Sentiment_Class"] == "Neutral").mean() * 100
avg_score = dff["Score"].mean()

# Inject card CSS once
st.markdown("""
<style>
.kpi-card {
    background: #1E1E1E;
    border: 1px solid #2E2E2E;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: left;
}
.kpi-label {
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1;
    color: #F0F0F0;
}
.kpi-value.positive { color: #1D9E75; }
.kpi-value.neutral  { color: #1A7FA0; }
.kpi-value.negative { color: #D48B2A; }
</style>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Total Reviews</div>
  <div class="kpi-value">{total:,}</div>
</div>""", unsafe_allow_html=True)

k2.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Positive Sentiment</div>
  <div class="kpi-value positive">{pct_pos:.1f}%</div>
</div>""", unsafe_allow_html=True)

k3.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Neutral Sentiment</div>
  <div class="kpi-value neutral">{pct_neu:.1f}%</div>
</div>""", unsafe_allow_html=True)

k4.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Negative Sentiment</div>
  <div class="kpi-value negative">{pct_neg:.1f}%</div>
</div>""", unsafe_allow_html=True)

k5.markdown(f"""
<div class="kpi-card">
  <div class="kpi-label">Avg Star Rating</div>
  <div class="kpi-value">{avg_score:.2f} / 5</div>
</div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")


# ==============================================================================
# ROW B — Monthly Sales Volume (full width)
# ==============================================================================
st.markdown("**Monthly Review Volume — Sales Velocity Proxy**")
st.caption("Review count per month as a proxy for sales activity. Consistent 8x growth from 2008 to 2012.")

monthly_vol = (
    dff.groupby("Year_Month").size()
    .reset_index(name="Review Count")
    .sort_values("Year_Month")
)

fig3 = px.area(
    monthly_vol, x="Year_Month", y="Review Count",
    labels={"Year_Month": "Month", "Review Count": "Review Count"},
    color_discrete_sequence=[COLORS["sales"]],
)
fig3.update_traces(
    line_color=COLORS["sales"],
    fillcolor="rgba(15, 110, 86, 0.15)",
)
fig3.update_xaxes(tickangle=40, nticks=22)
fig3 = style(fig3, height=260, mb=24)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")


# ============================================================================
# ROW B2 — Monthly Sentiment Polarity (leading indicator)
# ============================================================================
st.markdown("**Monthly Average Sentiment Polarity**")
st.caption("Use this to test whether drops in sentiment polarity appear before declines in review volume.")

monthly_sent = (
    dff.groupby("Year_Month")["Sentiment_Polarity"].mean()
    .reset_index()
    .sort_values("Year_Month")
)
monthly_sent["Rolling_Polarity"] = monthly_sent["Sentiment_Polarity"].rolling(window=3, min_periods=1).mean()

fig3b = px.line(
    monthly_sent,
    x="Year_Month",
    y=["Sentiment_Polarity", "Rolling_Polarity"],
    labels={"value": "Sentiment Polarity", "Year_Month": "Month", "variable": "Series"},
    color_discrete_map={
        "Sentiment_Polarity": COLORS["neutral"],
        "Rolling_Polarity": COLORS["positive"],
    },
)
fig3b.update_traces(mode="lines")
fig3b.update_xaxes(tickangle=40, nticks=22)
fig3b = style(fig3b, height=260, mb=24)
st.plotly_chart(fig3b, use_container_width=True)

st.markdown("---")


# ==============================================================================
# ROW B3 — Sentiment Polarity by Star Rating
# ==============================================================================
st.markdown("**Sentiment Polarity by Star Rating**")
st.caption("Distribution of sentiment polarity across 1–5 star ratings. Lower-star groups should skew lower in polarity.")

fig3c = px.box(
    dff,
    x="Score",
    y="Sentiment_Polarity",
    category_orders={"Score": [1, 2, 3, 4, 5]},
    labels={"Score": "Star Rating", "Sentiment_Polarity": "Sentiment Polarity"},
    points=False,
)
fig3c = style(fig3c, height=300, mb=24)
st.plotly_chart(fig3c, use_container_width=True)

st.markdown("---")


# ==================================``============================================
# ROW C — Product Sentiment vs Sales Scatter  |  Top 15 Products
# ==============================================================================
prod_agg = (
    dff.groupby(["ProductId", "Product_Display"])
    .agg(
        Sales_Volume=("Sales_Volume_Proxy", "max"),
        Avg_Sentiment=("Sentiment_Polarity", "mean"),
        Review_Count=("Id", "count"),
    )
    .reset_index()
)

colC1, colC2 = st.columns([1, 1])

with colC1:
    st.markdown("**Product Sentiment vs. Sales Volume**")
    st.caption("Do higher-sentiment products sell more? Each dot is one product.")

    fig4 = px.scatter(
        prod_agg,
        x="Avg_Sentiment", y="Sales_Volume",
        opacity=0.45,
        trendline="ols",
        color_discrete_sequence=[COLORS["positive"]],
        labels={
            "Avg_Sentiment": "Avg Sentiment Polarity",
            "Sales_Volume":  "Max Sales Volume Proxy",
        },
    )
    fig4.update_traces(marker=dict(size=5))
    fig4.data[1].line.color = COLORS["trend"]
    fig4.data[1].line.width = 2
    fig4 = style(fig4, height=340, mb=28, ml=52)
    st.plotly_chart(fig4, use_container_width=True)

with colC2:
    st.markdown("**Top 15 Highest Volume Products Colored by Average Sentiment**")
    st.caption("Bar height = max sales-volume proxy. Color = average sentiment polarity. Low-color, high-bar products may indicate quality risk.")

    prod_agg_eda8 = (
        dff.groupby("Product_Label")
        .agg({"Sentiment_Polarity": "mean", "Sales_Volume_Proxy": "max"})
        .reset_index()
    )
    top_prods = prod_agg_eda8.nlargest(15, "Sales_Volume_Proxy")

    fig5 = px.bar(
        top_prods,
        x="Product_Label",
        y="Sales_Volume_Proxy",
        color="Sentiment_Polarity",
        color_continuous_scale="RdYlGn",
        title="EDA 8: Top 15 Highest Volume Products Colored by Average Sentiment",
    )
    fig5.update_xaxes(tickangle=45)
    fig5 = style(fig5, height=340, mb=28)
    st.plotly_chart(fig5, use_container_width=True)

st.markdown("---")
st.caption("Dashboard · Streamlit + Plotly · Amazon Fine Food Reviews (Kaggle) · Sentiment: TextBlob")