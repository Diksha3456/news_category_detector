import gradio as gr
import joblib

model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def predict_category(article_text):
    if not article_text.strip():
        return "Please paste some article text."

    vec = vectorizer.transform([article_text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()

    return f"Category: {pred} (Confidence: {prob * 100:.1f}%)"


demo = gr.Interface(
    fn=predict_category,
    inputs=gr.Textbox(
        lines=10,
        placeholder="Yahan news article paste karo...",
        label="News Article"
    ),
    outputs=gr.Textbox(label="Predicted Category"),
    title="📰 News Article Category Detector",
    description="Article paste karo — batayega ki Sports, Politics ya Tech ka hai."
)

demo.launch()
