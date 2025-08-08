# Amazon Customer Segmentation and Recommendation 

# Project Overview
This project uses machine learning to analyze customer purchasing behaviour to identify customer sements. The model is built using a dataset from Kaggle and uses RFM metrics and clustering to discover product bundling techniques. The model alsp ouses market basket analysis to reveal high-confidence product associations.

# Features 
- Customer Segmentation
- Market Basket Analysis
- Interactive Dashboard
- Business Insights

# Technologies Used: Streamlit, Python 
- Machine Learning: Scikit-learn, K-Means, Apriori Algorithm
- Data Visualization: Matplotlib, Plotly, Seaborn
- Data Processing: Pandas, NumPy

# Installation 
Clone the repository:
```bash
git clone https://github.com/AyushmanShrestha2000/Amazon-Customer-Recommendation
cd amazon-customer-segmentation

# Install dependencies:
- pip install -r requirements.txt

# Run the application:
- streamlit run app.py

# File Structure
amazon-customer-segmentation/
├── app.py # Streamlit application
├── amazon.csv # Amazon sales dataset
├── README.md # This file
├── requirements.txt # Python dependencies
└── assets/ # Visualization assets

Data source: [Kaggle Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
