from transformers import pipeline

# ⭐ Load model once (global)
summarizer_model = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

def generate_summary(text):

    if len(text) < 200:
        return text

    result = summarizer_model(
        text[:1000],
        max_length=130,
        min_length=40,
        do_sample=False
    )

    return result[0]['summary_text']