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
    # Determine the domain based on the content of the prompt
    if "RCM" in prompt:
        domain = "RCM"
    elif "DHIS" in prompt:
        domain = "DHIS"
    else:
        domain = None  # Default if no domain-specific keyword is found
    
    with st.spinner("Generating response..."):
        # Pass the domain to run_llm
        generated_response = run_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"],
            domain=domain
        )
        formatted_response = f"{generated_response['answer']}"
        
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"] .append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response['answer']))


if st.session_state["chat_answers_history"]:
    for i, (generated_response, user_query) in enumerate(
        zip(st.session_state["chat_answers_history"], st.session_state["user_prompt_history"])
    ):
        # Define custom logos for user and AI
        user_logo = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTtuphMb4mq-EcVWhMVT8FCkv5dqZGgvn_QiA&s"  # Updated direct link
        ai_logo = "https://cdn-icons-png.flaticon.com/512/3774/3774299.png"

        # Display user query with custom logo
        message(user_query, is_user=True, logo=user_logo, key=f"user_message_{i}")

        # Display AI response without a logo
        message(generated_response, is_user=False, logo=ai_logo, key=f"ai_message_{i}")
