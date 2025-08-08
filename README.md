# Amazon Customer Segmentation and Recommendation System

## Project Overview
This project analyzes Amazon customer behavior using machine learning to identify valuable customer segments and product associations. It processes transaction data to create RFM (Recency, Frequency, Monetary) metrics, clusters customers, and discovers product bundling opportunities. The K-Means clustering achieved a 0.483 silhouette score, and market basket analysis revealed high-confidence product associations.

## Features 
- **Customer Segmentation**:
  - RFM analysis (Recency, Frequency, Monetary)
  - K-Means clustering with silhouette scoring
  - Hierarchical clustering comparison
- **Market Basket Analysis**:
  - Apriori algorithm implementation
  - Product association rules
  - Lift and confidence metrics
- **Interactive Dashboard**:
  - Cluster visualization
  - Customer segment profiles
  - Product recommendation engine
- **Business Insights**:
  - Segment-specific strategies
  - Revenue opportunity analysis
  - Implementation recommendations

## Technologies Used:
- **Frontend**: Streamlit
- **Backend**: Python
- **Machine Learning**:
  - Scikit-learn
  - K-Means
  - Apriori Algorithm
- **Data Visualization**:
  - Matplotlib
  - Seaborn
  - Plotly
- **Data Processing**: Pandas, NumPy

## Installation 
Clone the repository:
```bash
git clone https://github.com/AyushmanShrestha2000/Amazon-Customer-Recommendation
cd amazon-customer-segmentation

## Install dependencies:
- pip install -r requirements.txt

## Run the application:
- streamlit run app.py

## File Structure
amazon-customer-segmentation/
├── app.py # Streamlit application
├── amazon.csv # Amazon sales dataset
├── README.md # This file
├── requirements.txt # Python dependencies
└── assets/ # Visualization assets

Data source: [Kaggle Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
