# Forecasting-Regional-Conflict-in-Africa-Using-TensorFlow
_Collaborative Filtering Using TensorFlow to Analyze Regional Conflict in Africa_

This project implements a  Deep Learning model to forecast conflict intensity across the African continent. Using a Recursive Negative Binomial Regression approach, the system predicts the number of weekly conflict events for 54+ countries over an 8-week horizon.

## **Project Overview**
Predicting political violence is inherently difficult due to the "sparse" nature of the data. This project solves this by treating conflict forecasting as a Recommender System problem:
- Countries are "Users"
- Weeks are "Items"
- Event Counts are the "Ratings" we are trying to predict.

By using Collaborative Filtering via Neural Embeddings, the model learns latent patterns of violence that move across regions and time.

## **The Model: Deep Recursive Forecasting**
The core of this system is a Negative Binomial Neural Network built in TensorFlow. Unlike standard regression, this model handles the "over-dispersion" of conflict data (where most weeks are quiet, but some are extremely violent).

**Key Features:**
- Entity Embeddings: High-dimensional representations for countries and weeks to capture regional similarities.
- Recursive Lag Logic: The model uses its own $t+1$ prediction as the input for $t+2$, allowing for a multi-step look-ahead.
- Uncertainty Quantification: Optimized to minimize RMSLE (Root Mean Squared Logarithmic Error), ensuring the model isn't overly penalized for small misses in high-intensity zones.

## **Visualizing the Conflict Horizon**
The project includes visualization tools designed for policy-makers, researchers or other users wanting to view data from the model.

1. **Global Risk Matrix(The Heatmap)**: A 16-week "Master Matrix" that stitches together 8 weeks of historical truth with 8 weeks of AI-generated forecasts.
   - Blue Gradient: Historical Ground Truth.
   - Red Gradient: Predicted Volatility.
2. **Spatiotemporal Trend Analysis**: Individual country deep-dives that visualize the "Bridge" between past and future.
   - Solid Blue Line: Observed ACLED data.
   - Dashed Red Line: AI-projected trajectory.

## **Tech Stack**
- Backend: Python, TensorFlow 2.x, NumPyData
- Engineering: Pandas (Recursive Data Augmentation)
- Visualization: Matplotlib, Seaborn (Conditional Heatmaps)
- Interface: Gradio (Interactive Intelligence Dashboard)
- Data Source: ACLED (Armed Conflict Location & Event Data Project)

## **How to Use**
1. **Clone the Repo:** https://github.com/yourusername/africa-conflict-forecasting
2. **Install Dependencies:** pip install -r requirements.txt
3. **Run the Notebook:** Open Final_Project_Final_Notebook.ipynb to see the training process and the 16-week matrix generation.
4. **Launch the Dashboard:** Run the final cell to start the Gradio Intelligence Suite.

## **Disclaimer**
This model is intended for research and educational purposes. While it identifies statistical trends in political violence, it should not be used as the sole basis for security or humanitarian decisions.
