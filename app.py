import re
import joblib
import gradio as gr

# Load trained model and TF-IDF vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


# Text cleaning function
def clean_text(text):
    text = str(text)
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Keep only English letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Prediction function
def predict_news(news):
    if not news.strip():
        return "⚠️ Please enter a news article."

    cleaned_news = clean_text(news)

    news_tfidf = vectorizer.transform([cleaned_news])

    prediction = model.predict(news_tfidf)[0]

    if prediction == 0:
        return "❌ FAKE NEWS"
    else:
        return "✅ REAL NEWS"


# Gradio interface
demo = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Paste your news article here..."
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="📰 Fake News Detection",
    description="Enter a news article to predict whether it is Fake or Real."
)

demo.launch()
