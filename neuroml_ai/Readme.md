---
title: NeuroML-AI
sdk: docker
app_port: 7860
---

# NeuroML-AI

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

### Supported models

- local Ollama models: include the `ollama:` prefix.
- local vLLM OpenAI-compatible server: include the `vllm:` prefix. Set `VLLM_BASE_URL` if your server is not at `http://127.0.0.1:8000/v1`.
- HuggingFace inference providers: include a `huggingface:` prefix. You must obtain a HuggingFace token and set it in the `HF_TOKEN` environment variable.
