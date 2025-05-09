import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load model and tokenizer
MODEL_NAME = "nandha006/t5-small-finetuned-xsum"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

st.title("T5-small Finetuned Summarizer (xsum)")

st.write("Enter text to summarize using the finetuned T5 model.")
st.info("Note: Maximum input length is 256 tokens. The summary will be at most 60 tokens.")

input_text = st.text_area("Input Text", height=200)

if st.button("Generate Summary"):
    if input_text.strip() == "":
        st.warning("Please enter some text to summarize.")
    else:
        # Tokenize input
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=100, truncation=True)
        # Generate summary
        summary_ids = model.generate(inputs, max_length=60, min_length=10, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        st.subheader("Summary:")
        st.success(summary)
