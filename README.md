# Netflix-RecSys-
Netflix-Style Movie Recommendation Engine using Keras &amp; TensorFlow

# 🎬 Netflix-Style Movie Recommendation Engine

A collaborative filtering-based movie recommender system using deep learning built with TensorFlow & Keras.

##  Features

- Collaborative filtering using Embedding layers
- Trained on real movie rating data
- Predicts user ratings for unseen movies
- Top-N movie recommendations
- Evaluated using RMSE

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- TensorFlow + Keras

## 📊 Model Summary

Embedding size: 50  
Optimizer: Adam  
Loss: Mean Squared Error (MSE)  
Evaluation: Root Mean Squared Error (RMSE)



## 📁 Dataset

Ensure your dataset contains:
- `id` (user or viewer id)
- `original_title` (movie title)
- `vote_average` (user rating for that movie)

## 🧪 Usage

```python
# Train the model
model.fit(...)

# Recommend top 5 movies for user with ID 1
recommend_movies(user_id=1, df_original=df, model=model, top_n=5)

