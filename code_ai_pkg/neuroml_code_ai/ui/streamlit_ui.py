#!/usr/bin/env python3
"""
Streamlit chat app interface for RAG

File: gen_rag/ui/streamlit_ui.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio

import httpx
import streamlit as st
from neuroml_ai_utils.api import check_api_is_ready


def runner():
    """Main runner for streamlit app"""
    url = "http://127.0.0.1:8005"
    with st.spinner("Waiting for backend..."):
        try:
            asyncio.run(check_api_is_ready(f"{url}/health/ready"))
        except Exception as e:
            st.error(f"Could not connect to backend: {e}")
            st.stop

    st.title("NeuroML AI coding assistant")

    # get history and re-write it
    if "history" not in st.session_state:
        st.session_state.history = []

    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask anything", key="user"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            # stream = st.session_state.nml_ai.run_graph_stream(query)
            # response = st.write_stream(stream)
            with st.spinner("Working..."):
                with httpx.Client(timeout=None) as client:
                    response = client.post(f"{url}/query", params={"query": query})
                    response_result = response.json().get("result")
                    st.markdown(response_result)
        st.session_state.history.append(
            {"role": "assistant", "content": response_result}
        )


if __name__ == "__main__":
    runner()
