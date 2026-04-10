#  Gamesense — Smart Game Recommendation System

##  Overview

**Gamesense** is a machine learning-powered web application that helps users discover the best games using intelligent filters, personalized recommendations, and interactive data visualizations.

The system combines **content-based filtering** with **predictive modeling** to suggest similar games and identify hidden gems that users might otherwise overlook.

---

##  Features

###  Smart Filtering

* Filter games based on:

  * ⭐ Rating
  * 💰 Price
  * 🎲 Genres
  * 🖥️ Platforms
* Search games by name

---

###  Game Recommendations

* Uses **TF-IDF + Cosine Similarity**
* Suggests top 5 similar games based on:

  * Genres
  * Platforms
* Fast and dynamic (computed in real-time)

---

###  Predicted Ratings

* Uses **Ridge Regression model**
* Predicts game ratings based on:

  * Genres
  * Price
  * Popularity
* Helps users evaluate potential game quality

---

###  Hidden Gems Detection

* Identifies underrated games where:

  * Predicted rating > Actual rating
* Highlights games with strong potential but low visibility

---

###  Interactive Visualizations

*  Price vs Rating scatter plot
*  Top genres distribution chart
*  Smooth animations and UI effects

---

##  Machine Learning Models

### 1. TF-IDF Vectorizer

* Converts game features (genres + platforms) into numerical vectors

### 2. Cosine Similarity

* Measures similarity between games
* Used for recommendation system

### 3. MultiLabelBinarizer

* Encodes multiple genres into binary format for ML model

### 4. Ridge Regression

* Predicts game ratings
* Lightweight and efficient for deployment

---

##  Architecture

### 🔹 Data Processing (Notebook)

```
Raw Data → Cleaning → Feature Engineering → Model Training → Save .pkl files
```

### 🔹 Application Layer (Streamlit)

```
User Input → Filtering → Recommendation → Prediction → Visualization
```

---

##  Project Structure

```
Gamesense/
│
├── app.py                  # Streamlit application
├── requirements.txt        # Dependencies
├── cleaned_steam.csv       # Cleaned dataset
├── steam.csv               # Original dataset
├── steam.ipynb             # Notebook
├── assets/
│   ├── style.css           # Custom UI styling
│   ├── tfidf.pkl           # TF-IDF model
│   ├── rating_model.pkl    # ML model
│   ├── mlb.pkl             # Encoder
│   └── steam_processed.csv # Processed dataset
```

---

##  Installation & Setup

### 1. Clone Repository

```
git clone https://github.com/chetana12156/Gamesense.git
cd Gamesense
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Run Application

```
streamlit run app.py
```

---

##  Deployment

The app is deployed using **Streamlit Cloud**, which automatically installs dependencies and runs the application from the GitHub repository.

Live app : https://gamesense-2bf2hp65ydmjxons3eje3w.streamlit.app/

---

##  Key Highlights

*  Hybrid recommendation system (ML + content-based filtering)
*  Lightweight and optimized for deployment
*  Interactive and user-friendly UI
*  Real-time recommendations and predictions

---

##  Future Improvements

*  Game cards UI instead of tables
*  Advanced ML models (XGBoost / Neural Networks)
*  Search autocomplete
*  Personalized recommendations based on user behavior

---

##  Authors

**C.Vandana,**
**C.Chetana,**
**Ch.Preethi.**

---

##  Acknowledgements

* Steam dataset
* Scikit-learn
* Streamlit
* Plotly

---

##  Conclusion

Gamesense demonstrates how machine learning can be effectively combined with interactive web applications to create a smart and scalable recommendation system.

---
