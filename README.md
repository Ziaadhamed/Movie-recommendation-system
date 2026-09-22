# Movie Recommendation System

A movie recommendation system I built using the MovieLens 1M dataset (~1 million ratings from 6,000+ users on ~3,900 movies).

Originally built as a coursework project, cleaned up a bit for GitHub.

## What it does

I tried two different approaches to recommending movies:

- **Collaborative filtering** — used SVD (via the `surprise` library) to predict how a user would rate a movie based on other users with similar ratings.
- **Content-based filtering** — used TF-IDF on movie genres + cosine similarity to recommend movies similar to ones the user already liked.

I also experimented with a GAN (TensorFlow/Keras) to generate synthetic user profiles, mainly to see how it works and whether it could help with cold-start users (people with no rating history).

There's also a simple interactive part: you can enter a user ID and get recommendations, or if you're a "new user," rate one movie and get suggestions based on that.

## Tools used

- Python, pandas
- scikit-surprise (SVD)
- scikit-learn (TF-IDF, cosine similarity)
- TensorFlow/Keras (GAN part)
- MLflow (to log RMSE, precision/recall/F1 instead of just eyeballing results)

## How to run it

1. Unzip `Data.zip` — you need `movies.dat`, `ratings.dat`, and `users.dat` in the same folder as the notebook
2. Install the packages listed above
3. Run the notebook cells in order

Dataset source if you want a fresh copy: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## Results

- Collaborative filtering: RMSE and accuracy logged with MLflow
- Content-based filtering: Precision@10, Recall@10, F1@10 logged with MLflow

## Things I'd improve if I revisit this

- Combine both approaches into one ranking instead of keeping them separate
- Use more movie info (cast, director, year) instead of just genres for content-based filtering
- Build a simple UI instead of the input()-based interaction

---
Dataset: [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/), GroupLens Research.
