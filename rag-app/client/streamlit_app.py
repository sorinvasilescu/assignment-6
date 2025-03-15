import streamlit as st
import requests


def query_fastapi(query, top_k=5, max_tokens=200, temperature=0.7):
    """Send a query to the FastAPI backend and return the response."""
    url = "http://localhost:8000/generate"
    params = {
        "query": query,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


# Streamlit UI setup
st.set_page_config(page_title="Chat with FastAPI", layout="wide")
st.title("Chat Interface for FastAPI Backend")

# Initialize chat history if not already set
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input field
query = st.chat_input("Ask something...")
if query:
    # Display user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Get response from FastAPI backend
    response = query_fastapi(query)

    if "error" in response:
        answer = f"⚠️ Error: {response['error']}"
    else:
        answer = response.get("answer", "No response from server.")

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Store response in session state
    st.session_state["messages"].append({"role": "assistant", "content": answer})
