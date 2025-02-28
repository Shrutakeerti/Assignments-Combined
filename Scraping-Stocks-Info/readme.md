# Stock Movement Prediction Using Reddit Sentiment Analysis

## Project Overview
This project predicts stock movements based on sentiment analysis of Reddit posts from subreddits like `r/stocks`, `r/wallstreetbets`, and `r/investing`. It uses **FinBERT** for sentiment analysis and a **Random Forest** model for predicting stock movements.

**Current Model Accuracy**: 0.87

---

## Setup Requirements

### 1. Clone the repository:
```bash
git clone https://github.com/your_username/your_repository.git
cd your_repository

```
## 2. Install Dependencies

### Ensure you have **Python 3.8+** installed. Create a virtual environment and install the required libraries:

### 1. Run the following command to create a virtual environment:
   ```bash
   python -m venv venv

   ```
### 2. On macOS/Linux:
    
    source venv/bin/activate
    
    
### 3.On Windows:  

     
    venv\Scripts\activate

    
## Install the Required Libraries

Ensure you have **Python 3.8+** installed. Install the required libraries:

```bash
pip install -r requirements.txt
```


## Add Reddit API Credentials

Create a Reddit app [here](https://www.reddit.com/prefs/apps) and add your credentials in the code:

```python
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID', 
                     client_secret='YOUR_CLIENT_SECRET', 
                     user_agent='YOUR_USER_AGENT')
```


## How to Run the Project

1. **Scrape data from Reddit:**
   ```bash
   python scrape_reddit.py
   ```
2. **Perform sentiment analysis using FinBERT:**
   ```bash
   python sentiment_analysis.py

   ```
3. **Train the stock movement prediction model:**
   ```bash
   python train_model.py
   ```
4. **Evaluate the Model:**

```bash
python evaluate_model.py

```
