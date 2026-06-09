---
title: NeuroML AI Assistant Framework
sdk: docker
app_port: 7860
---

# NeuroML AI Assistant Framework

AI assistant for helping with NeuroML queries and model generation.

![Langchain schematic](nml-ai-assistant-lang-graph.png "Langchain schematic")

Please note that this project is under active development and does not currently provide a stable release/API/ABI.

## Usage

Install the package and dependencies using `pip` or `uv pip` from the GitHub repository:

```
# in the `utils_pkg` folder:
pip install .

# in the `neuroml_ai` folder:
pip install .
```

Start the API server:

```
fastapi dev neuroml_ai/api/main.py --port 8005
```

The following environment variables need to be set:

- `GEN_RAG_CHAT_MODEL`: the name of the chat model to use. See below.
- `GEN_RAG_VS_CONFIG`: the path to the configuration file for the vector stores.
