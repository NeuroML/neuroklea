#!/usr/bin/env python3
"""
LLM related utils

File: gen_rag/llm.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import os
import re
import sys
import time
from functools import lru_cache
from pathlib import Path
from textwrap import dedent
from typing import Optional, Type, Union

import ollama
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFaceEndpoint,
    HuggingFaceEndpointEmbeddings,
)
from pydantic import BaseModel

logging.basicConfig(
    format="%(name)s (%(levelname)s) >>> %(message)s\n", level=logging.WARNING
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def check_ollama_model(logger, model, exit=False):
    """Check if ollama model is available

    :param logger: logger instance
    :type logger: logging
    :param model: ollama model name
    :type model: str
    :param exit: if we should call sys.exit if check fails
    :type exit: bool
    :returns: None

    :throws ollama.ResponseError: if `model` is not available
    :throws ConnectionError: if cannot connect to an Ollama server

    """
    try:
        _ = ollama.show(model)
    except ollama.ResponseError:
        logger.error(f"Could not find ollama model: {model}")
        logger.error("Please ensure you have pulled the model")
        if exit:
            sys.exit(-1)
    except ConnectionError:
        logger.error("Could not connect to Ollama.")
        if exit:
            sys.exit(-1)


def parse_output_with_thought[TSchema: BaseModel](
    message: AIMessage, schema: Type[TSchema]
) -> TSchema:
    """Parse AI message with thought to a dict based on given schema"""
    if "</think>" in message.content:
        splits = message.content.split("</think>")
        answer = splits[1].strip()
    else:
        answer = message.content

    parser = JsonOutputParser()
    parser.pydantic_object = schema()
    result = parser.parse(answer)
    return result


def split_output_by_section(
    text: str, section_start_marker: str, section_end_marker: Optional[str] = None
):
    """Split out thoughts and actual responses from AI responses"""
    if not text:
        logger.warning("Empty message.content. Nothing to do.")
        return "", ""

    if not section_start_marker:
        logger.warning("No starting marker. Nothing to do.")
        return "", ""

    if not section_end_marker:
        section_end_marker = None

    delimited, other = [], []

    # prepare pattern
    markers = [re.escape(section_start_marker)]
    if section_end_marker:
        markers.append(re.escape(section_end_marker))
    pattern = f"({'|'.join(markers)})"

    # split
    splits = re.split(pattern, text)

    # process splits
    # by default, we're outside the delimiters to begin with
    is_in = False

    # do we have both markers?
    found_start_marker = section_start_marker in text
    found_end_marker = section_end_marker in text if section_end_marker else False

    if not found_start_marker and not found_end_marker:
        logger.debug("No markers found. Nothing to do.")
        return "", text

    # end marker, but no start: we start inside the delimted region
    if found_end_marker and not found_start_marker:
        is_in = True

    for part in splits:
        if part == section_start_marker:
            is_in = True
        elif part == section_end_marker:
            is_in = False
        else:
            if is_in:
                delimited.append(part)
            else:
                other.append(part)

    delimited_text = "".join(delimited).strip()
    other_text = "".join(other).strip()

    # Add notes
    if found_start_marker and (section_end_marker and not found_end_marker):
        other_text += "\nNOTE: NO END MARKER FOUND"
    elif found_end_marker and not found_start_marker:
        other_text += "\nNOTE: NO START MARKER FOUND"

    return delimited_text, other_text


def check_model_works(model, timeout=30, retries=5):
    """Check if a model works since it is not tested when loaded"""
    assert timeout >= 0

    for attempt in range(retries):
        print(f"Checking model. Attempt #{attempt + 1}/{retries}")
        try:
            # Use a very simple prompt with short max_tokens
            result = model.invoke("Hello world", config={"timeout": timeout})
            print(f"Model available (attempt {attempt + 1}/{retries}): {result}")
            return True, f"Model available (attempt {attempt + 1}/{retries})"
        except StopIteration as e:
            return (
                False,
                f"{e.__class__.__name__}: check if any inference providers are available for the selected model",
            )
        except Exception as e:
            error_msg = f"{e.__class__.__name__}: {e.__str__()}"
            print(f"Attempt #{attempt + 1}/{retries}: model unavailable: {error_msg}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print(f"Model unavailable after {retries} attempts: {error_msg}")
                return (
                    False,
                    f"Model unavailable after {retries} attempts: {error_msg}",
                )
            error_msg = f"{e.__class__.__name__}: {e.__str__()}"
            print(f"Attempt #{attempt}: model unavailable: {error_msg}")
            if attempt < retries - 1:
                time.sleep(2**attempt)  # Exponential backoff
            else:
                print(f"Model unavailable after {retries} attempts: {error_msg}")
                return (
                    False,
                    f"Model unavailable after {retries} attempts: {error_msg}",
                )
    return False, "Unknown error"


def setup_embedding(model_name_full, logger):
    # need to use inference providers
    if model_name_full.lower().startswith("huggingface:"):
        _, model_name, provider = model_name_full.split(":")
        logger.debug(f"Using huggingface model: {model_name}")

        hf_token = os.environ.get("HF_TOKEN", None)
        assert hf_token

        model_var = HuggingFaceEndpointEmbeddings(
            model=f"{model_name}",
            provider="auto",
            task="feature-extraction",
            huggingfacehub_api_token=hf_token,
        )
    else:
        if model_name_full.lower().startswith("ollama:"):
            check_ollama_model(logger, model_name_full.lower().replace("ollama:", ""))
        model_var = init_embeddings(model_name_full)

    assert model_var
    logger.info(f"Using embedding model: {model_name_full}")
    return model_var


def setup_llm(model_name_full: str, logger: logging.Logger):
    """Set up a chat model"""
    if model_name_full.lower().startswith("huggingface:"):
        _, model_name, provider = model_name_full.split(":")
        logger.debug(f"Using huggingface model: {model_name}")

        hf_token = os.environ.get("HF_TOKEN", None)
        assert hf_token

        logger.debug("Got HuggingFace Token")

        llm = HuggingFaceEndpoint(
            repo_id=f"{model_name}",
            provider="auto",
            max_new_tokens=32768,
            do_sample=False,
            repetition_penalty=1.03,
            task="conversational",  # seems to be ignored, defaults to text-generation
            huggingfacehub_api_token=hf_token,
        )

        model_var = ChatHuggingFace(llm=llm)

        """

        model_var = init_chat_model(
            model_name,
            model_provider="huggingface",
            llm=llm,
            configurable_fields=("temperature"),
            backend="endpoint",
        )
        """
        assert model_var

        state, msg = check_model_works(model_var, timeout=10, retries=3)
        if state:
            logger.debug(f"Model works: {state}, {msg}")
        else:
            logger.debug(f"Model does not work: {state}, {msg}")
        assert state
    else:
        if model_name_full.lower().startswith("ollama:"):
            check_ollama_model(logger, model_name_full.lower().replace("ollama:", ""))

        model_var = init_chat_model(
            model_name_full, configurable_fields=("temperature")
        )
        assert model_var

        state, msg = check_model_works(model_var, timeout=60)
        assert state

    logger.info(f"Using chat model: {model_name_full}")

    return model_var


def get_last_n_conversations(
    all_messages, start: int = 0, stop: Optional[int] = None
) -> tuple[str, list[HumanMessage], list[AIMessage]]:
    """Get recent converstations between start and stop indices

    :param all_messages: all the messages
    :param start: start index
    :param stop: stop index
    :returns: (conversation, list of human messages, list of ai messages)

    """
    conv_messages = list(
        filter(
            lambda x: isinstance(x, (HumanMessage, AIMessage)),
            all_messages[start:stop],
        )
    )
    human_messages = []
    ai_messages = []
    conversation = ""
    for msg in conv_messages:
        if isinstance(msg, HumanMessage):
            conversation += f"{msg.pretty_repr()}"
            human_messages.append(msg)
        else:
            conversation += f": {msg.pretty_repr()}"
            ai_messages.append(msg)

    print(f"{conversation = }")

    return (
        conversation.replace("{", "{{").replace("}", "}}"),
        human_messages,
        ai_messages,
    )


def get_history_summary_prompt(
    conversation: str, current_summary: str, logger: logging.Logger
):
    """Get a prompt for a history summary

    :param conversation: conversation to summarise
    :param current_summary: summary so far
    :param logger: logger object
    :returns: prompt to pass to LLM

    """
    # Summarise history
    system_prompt = dedent("""You are a memory/conversation summarisation
    assistant. Your job is to maintain a concise, factual memory of an
    ongoing conversation between a user and an AI assistant. This history
    will help the AI assistant in future conversations with the user.

    Guidelines:

    1. Preserve key facts, user intentions, user requirements, and user
    constraints.
    2. Remove filler, greetings, and irrelevant small talk.
    3. Keep the summary coherent and readable as a standalone record.
    4. Exclude reasoning steps, or internal thought processes. Do not add
    explanations or commentary. Exclude requests to summarise the
    conversation in the summary.
    5. Limit the summary to 5-10 sentences
    unless the conversation is very complex.
    6. Make it self-contained. Clearly note what the user said, and what the assistant's reply was.

    """)

    user_prompt = dedent("""
    Please create a summary of the conversation between the user and the AI
    assistant.

    ------

    Here is the current summary of the conversation so far:

    {old_summary}

    ------

    Here are the exchanges between the user and the assistant since the
    last summarisation:

    {conversation}

    """)

    prompt_template = ChatPromptTemplate(
        [("system", system_prompt), ("human", user_prompt)]
    )

    prompt = prompt_template.invoke(
        {
            "old_summary": current_summary,
            "conversation": conversation,
        }
    )
    logger.debug(f"{prompt =}")

    return prompt


def add_memory_to_prompt(context_summary: str, messages, num_recent_messages) -> str:
    """Add memory to system prompt.

    Adds the context summary and recent conversation

    :param state: agent state
    :returns: "memory" string to add to the system prompt

    """
    ret_string = ""

    directive = dedent("""
        IMPORTANT:

        - Consider both the latest user message AND the conversation history.

    """)

    if len(context_summary):
        ret_string += dedent(f"""
        -----

        Here is a concise summary of the previous conversation to maintain
        continuity:

        {context_summary}

        -----
        """)

    conversation, _, _ = get_last_n_conversations(
        messages, (-1 * num_recent_messages), None
    )
    if len(conversation):
        ret_string += dedent(f"""
        -----

        Here are the recent messages between the user and the assistant:

        {conversation}

        -----
        """)

    if len(ret_string):
        ret_string += directive

    return ret_string


@lru_cache(maxsize=10000)
def load_prompt(prompt_name: str, prompt_registry_location: str):
    """Load a prompt from file called prompt_name.md

    :param str: prompt file name
    :param prompt_registry_location: location of prompts folder/registry
    :returns: loaded prompt text

    """
    prompt_path = Path(f"{prompt_registry_location}/{prompt_name}.md")
    if not prompt_path.exists():
        raise FileNotFoundError(f"{prompt_path} was not found")

    with open(prompt_path, "r") as f:
        return f.read()
