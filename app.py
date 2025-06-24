import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model dari Hugging Face
@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("Ricky131/model-hoax-gpt2")
    tokenizer = AutoTokenizer.from_pretrained("Ricky131/model-hoax-gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

# Fungsi generator
def generate_hoax(input_text, max_length=40):
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# UI
st.title("🤖 Generator Teks Hoax Bahasa Indonesia")
prompt = st.text_input("Masukkan prompt awal:", "")

if st.button("Generate"):
    with st.spinner("Generating..."):
        result = generate_hoax(prompt)
        st.success("Hasil:")
        st.write(result)
