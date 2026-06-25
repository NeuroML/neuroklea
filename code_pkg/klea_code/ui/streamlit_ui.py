#!/usr/bin/env python3
"""
Streamlit chat app interface for code

File: klea_code/ui/streamlit_ui.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import sys
import uuid

import httpx
import streamlit as st
from klea_utils.api import check_api_is_ready


def runner():
    """Main runner for streamlit app"""
    title = sys.argv[1]
    url = sys.argv[2]
    try:
        with st.spinner("Waiting for backend..."):
            asyncio.run(check_api_is_ready(f"{url}/health/ready"))
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
        st.stop()

    st.title(title)
    st.info(
        "The answers are generated using an LLM. They may be inaccurate.  Please check with the documentation at https://docs.neuroml.org."
    )

    # get history and re-write it
    if "history" not in st.session_state:
        st.session_state.history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    for i, message in enumerate(st.session_state.history):
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if query := st.chat_input("Ask anything", key="user"):
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Working..."):
                response_result = ""
                with httpx.Client(timeout=None) as client:
                    try:
                        response = client.post(
                            f"{url}/query",
                            json={
                                "query": query,
                                "session_id": st.session_state.session_id,
                            },
                        )
                        response_result = response.json().get("result")
                        st.markdown(response_result)
                    except httpx.RequestError as e:
                        st.error("An error occured. Please try again")
                        st.error(f"```\n{e}\n```")
                st.session_state.history.append(
                    {"role": "assistant", "content": response_result}
                )


if __name__ == "__main__":
    runner()
