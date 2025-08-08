import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# For clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

# For association rules
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# For visualization
import plotly.express as px
import plotly.graph_objects as go

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

# Streamlit config
st.set_page_config(page_title="Amazon Customer Analytics", layout="wide")

# Load main data function with caching
@st.cache_data
def load_data():
    df = pd.read_csv('amazon.csv')
    
    # Data cleaning (same as original)
    cols_to_drop = ['about_product', 'review_title', 'review_content', 'img_link',
                    'product_link', 'user_name', 'review_id']
    df = df.drop(columns=cols_to_drop, errors='ignore')
    df.dropna(inplace=True)
    
    # Convert price columns
    df['discounted_price'] = df['discounted_price'].replace('[₹,]', '', regex=True).astype(float)
    df['actual_price'] = df['actual_price'].replace('[₹,]', '', regex=True).astype(float)
    
    # Create transaction date
    try:
        df['transaction_date'] = pd.to_datetime(df['review_date'])
        df = df.drop(columns=['review_date'], errors='ignore')
    except:
        df['transaction_date'] = pd.date_range(end=pd.Timestamp.today(), periods=len(df), freq='D')
    
    df['monetary_value'] = df['discounted_price']
    return df

# RFM calculation function
@st.cache_data
def calculate_rfm(df):
    reference_date = df['transaction_date'].max() + timedelta(days=1)
    rfm = df.groupby('user_id').agg({
        'transaction_date': lambda x: (reference_date - x.max()).days,
        'product_id': 'count',
        'monetary_value': 'sum'
    }).reset_index()
    rfm.columns = ['user_id', 'recency', 'frequency', 'monetary']
    rfm['avg_order_value'] = rfm['monetary'] / rfm['frequency']
    return rfm

# Clustering function with error handling for association rules
@st.cache_data
def perform_analysis(df, n_clusters=4, min_support=0.001):  # Lower default min_support
    # RFM calculation
    rfm = calculate_rfm(df)
    
    # Clustering
    features = ['recency', 'frequency', 'monetary', 'avg_order_value']
    X = rfm[features]
    
    # Handle outliers
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    mask = ~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)
    X_clean = X[mask]
    rfm_clean = rfm[mask].copy()
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=42)
    rfm_clean['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Market basket analysis with error handling
    transactions = df.groupby(['user_id', 'product_id'])['product_id'].count().unstack().fillna(0)
    transactions = transactions.applymap(lambda x: 1 if x > 0 else 0)
    
    try:
        frequent_itemsets = apriori(transactions, min_support=min_support, use_colnames=True)
        if len(frequent_itemsets) == 0:
            st.warning("No frequent itemsets found with current parameters. Try lowering the minimum support.")
            return rfm_clean, None
        
        rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
    except Exception as e:
        st.error(f"Error in market basket analysis: {str(e)}")
        return rfm_clean, None
    
    return rfm_clean, rules

def plot_cluster_analysis(rfm_clean, features):
    # Visualize clusters using PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(rfm_clean[features])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=rfm_clean['Cluster'], cmap='viridis', alpha=0.6)
    ax1.set_xlabel('PCA Component 1')
    ax1.set_ylabel('PCA Component 2')
    ax1.set_title('Customer Segments Visualization')
    plt.colorbar(scatter, ax=ax1, label='Cluster')
    
    # Cluster profiles radar chart
    cluster_summary = rfm_clean.groupby('Cluster')[features].mean()
    cluster_profiles_normalized = (cluster_summary - cluster_summary.min()) / (cluster_summary.max() - cluster_summary.min())
    
    categories = features
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]
    
    fig2, ax2 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow']
    
    for idx, row in cluster_profiles_normalized.iterrows():
        values = row.values.flatten().tolist()
        values += values[:1]
        ax2.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {idx}', color=colors[idx])
        ax2.fill(angles, values, alpha=0.15, color=colors[idx])
    
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories)
    ax2.set_title('Normalized RFM Cluster Profiles', size=16, y=1.1)
    ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    return fig1, fig2

def plot_association_rules(rules):
    if rules is None or len(rules) == 0:
        return None
    
    fig = plt.figure(figsize=(12, 8))
    scatter = plt.scatter(rules['support'], rules['confidence'],
                         c=rules['lift'], s=rules['lift']*20,
                         alpha=0.6, cmap='viridis')
    plt.xlabel('Support')
    plt.ylabel('Confidence')
    plt.title('Association Rules: Support vs Confidence')
    plt.colorbar(scatter, label='Lift')
    plt.grid(True, alpha=0.3)
    return fig

def main():
    st.title("Amazon Customer Segmentation & Market Basket Analysis")
    st.markdown("---")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
    
    # Sidebar controls
    with st.sidebar:
        st.header("Analysis Parameters")
        n_clusters = st.slider("Number of clusters", 2, 7, 4)
        min_support = st.slider("Minimum support for association rules", 0.0001, 0.1, 0.001, 0.0001, format="%.4f")
        
        st.markdown("---")
        st.header("Customer Lookup")
        customer_id = st.selectbox("Select customer ID", df['user_id'].unique())
        
        st.markdown("---")
        st.header("Data Info")
        st.write(f"Total transactions: {len(df)}")
        st.write(f"Unique customers: {df['user_id'].nunique()}")
        st.write(f"Unique products: {df['product_id'].nunique()}")
    
    # Perform analysis
    with st.spinner("Analyzing data..."):
        rfm_clean, rules = perform_analysis(df, n_clusters, min_support)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["Customer Segmentation", "Market Basket Analysis", "Business Insights"])
    
    with tab1:
        st.header("Customer Segmentation Results")
        
        if rfm_clean is not None:
            features = ['recency', 'frequency', 'monetary', 'avg_order_value']
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cluster Distribution")
                cluster_counts = rfm_clean['Cluster'].value_counts().sort_index()
                fig_counts = px.bar(cluster_counts, 
                                  x=cluster_counts.index, 
                                  y=cluster_counts.values,
                                  labels={'x':'Cluster', 'y':'Count'},
                                  color=cluster_counts.index)
                st.plotly_chart(fig_counts, use_container_width=True)
            
            with col2:
                st.subheader("Cluster Characteristics")
                cluster_summary = rfm_clean.groupby('Cluster')[features].mean()
                st.dataframe(cluster_summary.style.background_gradient(cmap='Blues'))
            
            st.subheader("Cluster Visualization")
            fig1, fig2 = plot_cluster_analysis(rfm_clean, features)
            st.pyplot(fig1)
            st.pyplot(fig2)
        else:
            st.error("No clustering results available")
    
    with tab2:
        st.header("Market Basket Analysis")
        
        if rules is not None and len(rules) > 0:
            st.subheader("Top Association Rules")
            st.dataframe(rules.sort_values('lift', ascending=False).head(10))
            
            st.subheader("Rules Visualization")
            fig_rules = plot_association_rules(rules)
            if fig_rules:
                st.pyplot(fig_rules)
            
            st.subheader("High-Quality Rules (Lift > 2 & Confidence > 0.5)")
            high_quality_rules = rules[(rules['lift'] > 2) & (rules['confidence'] > 0.5)]
            if len(high_quality_rules) > 0:
                st.dataframe(high_quality_rules)
            else:
                st.warning("No high-quality rules found with current parameters")
        else:
            st.warning("No association rules generated. Try adjusting the minimum support.")
    
    with tab3:
        st.header("Business Insights & Recommendations")
        
        if rfm_clean is not None:
            # Cluster personas
            cluster_personas = {
                0: {"name": "High-Value Loyalists", "strategy": "Offer VIP discounts and early access to new products"},
                1: {"name": "At-Risk Customers", "strategy": "Win-back campaigns with special offers"},
                2: {"name": "Budget Shoppers", "strategy": "Bundle deals and value packs"},
                3: {"name": "Occasional Buyers", "strategy": "Personalized recommendations based on past purchases"}
            }
            
            st.subheader("Customer Segment Strategies")
            for cluster_id in sorted(rfm_clean['Cluster'].unique()):
                persona = cluster_personas.get(cluster_id, {"name": f"Cluster {cluster_id}", "strategy": "General marketing"})
                with st.expander(f"{persona['name']} (Cluster {cluster_id})"):
                    st.markdown(f"**Recommended Strategy:** {persona['strategy']}")
                    
                    # Get cluster stats
                    cluster_data = rfm_clean[rfm_clean['Cluster'] == cluster_id]
                    st.markdown(f"**Segment Size:** {len(cluster_data)} customers ({len(cluster_data)/len(rfm_clean)*100:.1f}%)")
                    
                    # Show metrics
                    cols = st.columns(4)
                    metrics = ['recency', 'frequency', 'monetary', 'avg_order_value']
                    for i, metric in enumerate(metrics):
                        cols[i].metric(
                            label=metric.capitalize(),
                            value=f"{cluster_data[metric].mean():.1f}",
                            delta=f"{cluster_data[metric].median():.1f} median"
                        )
            
            if rules is not None and len(rules) > 0:
                st.subheader("Product Bundling Opportunities")
                top_rules = rules.sort_values('lift', ascending=False).head(3)
                for idx, rule in top_rules.iterrows():
                    antecedents = ", ".join(list(rule['antecedents']))
                    consequents = ", ".join(list(rule['consequents']))
                    st.markdown(f"""
                    **Rule {idx+1}**  
                    If customers buy: *{antecedents}*  
                    Then recommend: *{consequents}*  
                    Confidence: {rule['confidence']:.2f}, Lift: {rule['lift']:.2f}
                    """)

if __name__ == "__main__":
    main()
