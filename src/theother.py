from transformers import pipeline

def summarize_text(text):
    # Create a summarization pipeline
    summarizer = pipeline("summarization")

    # Generate summary
    summary = summarizer(text, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    return summary[0]['summary_text']

# Example text for summarization
input_text = """
The Hugging Face Transformers library provides easy-to-use interfaces for working with state-of-the-art natural language processing models.
It includes pre-trained models for various tasks, including text summarization. In this example, we'll build a simple summarization pipeline
using the Transformers library to generate a summary of a given input text.
"""

# Call the summarization function
summarized_text = summarize_text(input_text)

# Print the original and summarized text
print("\nSummarized Text:\n", summarized_text)
