import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to model files in Google Drive
model_path = '/content/drive/MyDrive/customer_support_bot'  

# Load the model and tokenizer from Google Drive
model = AutoModelForCausalLM.from_pretrained(model_path, revision="main")
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Define the prediction function
def predict(ticket_description):
    inputs = tokenizer(ticket_description, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model.generate(**inputs, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


# Streamlit UI
st.title("Customer Support Ticket Assistant Bot")

ticket_description = st.text_area("Enter Ticket Description:", height=200)

if st.button("Generate Response"):
    if ticket_description:
        response = predict(ticket_description)
        st.write("Bot Response:", response)
    else:
        st.write("Please enter a ticket description.")
