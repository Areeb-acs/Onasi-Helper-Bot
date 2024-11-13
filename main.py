from backend.core import run_llm
import streamlit as st
from streamlit_chat import message
import os

st.header("Onasi Helper Bot")

prompt = st.text_input("Prompt", placeholder="Enter your prompt here...")


if (
    "chat_answers_history" not in st.session_state
    and "user_prompt_history" not in st.session_state
    and "chat_history" not in st.session_state
):
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []




if prompt: 
    with st.spinner("Generating response..."):
        
        generated_response = run_llm(query=prompt, chat_history=st.session_state["chat_history"])
        formatted_response = f"{generated_response['answer']}"
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"] .append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response['answer']))


if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(
        zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"])
    ):
        # Add unique keys for each message
        message(user_query, is_user=True, key=f"user_message_{i}")
        message(generated_response, is_user=False, key=f"ai_message_{i}")