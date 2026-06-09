#!/usr/bin/env python3
"""
Tools for generating NeuroML code

File: codegen_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import asyncio
import logging
import sys
from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict

import aiohttp
from cachetools import TTLCache
from fastmcp import Context
from neuroml_ai_utils.plogging import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ..utils import ToolInfo, tool_meta

# set the implementation for development
from .sandbox import nml_mcp_sandbox
from .sandbox.sandbox import RunCommand
from .web_tools import _download_file_to_cache_by_content

sbox = nml_mcp_sandbox

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)
stdout_handler.addFilter(LoggerInfoFilter())
stdout_handler.setFormatter(logger_formatter_info)
logger.addHandler(stdout_handler)

stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.DEBUG)
stderr_handler.addFilter(LoggerNotInfoFilter())
stderr_handler.setFormatter(logger_formatter_other)
logger.addHandler(stderr_handler)


MAX_RESULTS = 20
SEARCH_TIMEOUT = aiohttp.ClientTimeout(total=30)
XML_DOWNLOAD_TIMEOUT = aiohttp.ClientTimeout(total=60)

# Cache for search results (2 hour TTL, max 100 entries)
NEUROMLDB_SEARCH_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)

# Cache for XML downloads (2 hour TTL, max 100 entries)
NEUROMLDB_XML_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)

# OSBv2 cache
OSBv2_SEARCH_CACHE: TTLCache[str, Any] = TTLCache(maxsize=100, ttl=7200)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True,
)
async def _search_neuromldb(session, url, query):
    """Search NeuroML-DB with retry logic."""
    r = await session.get(
        url,
        params={"q": query},
        timeout=SEARCH_TIMEOUT,
        ssl=False,
        raise_for_status=True,
    )
    async with r:
        return await r.json(content_type=None)


@retry(
    wait=wait_random_exponential(multiplier=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
    reraise=True,
)
async def _search_osbv2_repos(session, url, query, content_types, user_id, max_num):
    """Search NeuroML-DB with retry logic."""
    r = await session.get(
        url,
        params={
            "q": query,
            "types": content_types,
            "user_id": user_id,
            "page": 1,
            "per_page": max_num,
        },
        timeout=SEARCH_TIMEOUT,
        ssl=False,
        raise_for_status=True,
    )
    logger.debug(f"{r.request_info = }")
    async with r:
        return await r.json(content_type=None)


@tool_meta(ToolInfo(tags={"testing", "neuroml"}))
async def dummy_tool(astring: str) -> str:
    """Return the input string in a sentence (testing tool only).

    This is a dummy tool used only for unit testing and debugging.
    It simply returns the input string in a formatted sentence.

    Do NOT use in production workflows - this tool provides no real functionality.

    Args:
        astring: Any string to be echoed back

    Returns:
        The input string formatted as "I got {astring}"

    Example:
        result = dummy_tool("hello")
        # Returns: "I got hello"
    """
    return f"I got {astring}"


@tool_meta(ToolInfo(tags={"testing", "neuroml"}))
def create_new_NeuroML_model_tool(model_name: str = "NeuroMLModel") -> str:
    """Create a new blank NeuroML model template.

    Use this tool to generate a starting template for NeuroML models.
    This creates the basic structure that you can then customize with
    cells, networks, and simulations.

    Common use cases:
    - Starting a new NeuroML project
    - Creating a template for educational models
    - Generating boilerplate for complex networks

    Do NOT use for:
    - Reading existing NeuroML files (use file reading tools)
    - Validating existing models (use validation tools)
    - Running simulations (use simulation tools)

    Args:
        model_name: Name for the NeuroML model. Will be used as the network ID.
                  Spaces will be automatically removed for XML compatibility.
                  Defaults to "NeuroMLModel" if not specified.

    Returns:
        String containing Python code template that:
        - Imports necessary NeuroML utilities
        - Creates a NeuroMLDocument
        - Adds a Network with the specified name
        - Is ready for further customization

    Example:
        # Create a basic model template
        template = create_new_NeuroML_model_tool("MyNeuralNetwork")

        # Result can be executed with run_python_code_tool()
        # to create the actual NeuroML document structure

    Template structure:
    The generated code will create:
    - A NeuroMLDocument with unique ID
    - A Network with the specified model_name
    - Proper imports and setup for further development

    Next steps after using this tool:
    1. Execute the generated code with run_python_code_tool()
    2. Add cells, populations, and projections to the network
    3. Define simulation parameters
    4. Export to NeuroML XML format
    """
    model_name = model_name.replace(" ", "")

    model_str = dedent(
        f"""
    from neuroml.utils import component_factory

    nml_document = component_factory(neuroml.NeuroMLDocument, id="{model_name}")
    network = nml_document.add(neuroml.Network, id="{model_name}", validate=False)

    """
    )

    return model_str


@tool_meta(ToolInfo(tags={"testing", "neuroml"}))
async def run_lems_simulation(lems_file: str) -> Dict[str, Any]:
    """Execute a LEMS simulation using pynml and jLEMS simulator.

    Use this tool to run NeuroML simulation files and generate results.
    This is the standard way to execute LEMS (Low Entropy Model Specification)
    simulations for NeuroML models.

    Common use cases:
    - Running NeuroML model simulations
    - Testing model behavior with different parameters
    - Generating simulation outputs (traces, plots)
    - Validating model dynamics

    Do NOT use for:
    - Creating LEMS files (use NeuroML generation tools)
    - Running other types of simulations (use specific tools)
    - Analyzing results (use data analysis tools)

    Prerequisites:
    - The LEMS file must exist and be valid XML
    - pynml must be installed in the environment
    - jLEMS simulator must be available
    - All referenced NeuroML files must be accessible

    Args:
        lems_file: Path to the LEMS simulation XML file. Can be relative or absolute.
                  Must be a valid LEMS file that references existing NeuroML models.
                  File extension should typically be .xml.

    Returns:
        Dictionary with simulation execution results:
        - stdout: Simulation output, including progress messages and results
        - stderr: Error messages, warnings, or debug information
        - returncode: 0 for successful simulation, non-zero for errors
        - data: Additional metadata (execution time, resource usage, etc.)

    Examples:
        # Run a basic simulation
        result = await run_lems_simulation("LEMS_NML2_Ex9_Dynamics.xml")

        # Run simulation in subdirectory
        result = await run_lems_simulation("./simulations/basic_sim.xml")

        # Check if simulation was successful
        if result["returncode"] == 0:
            print("Simulation completed successfully")
            print(f"Output: {result['stdout']}")
        else:
            print(f"Simulation failed: {result['stderr']}")

    Simulation process:
    1. Validates LEMS file syntax
    2. Loads referenced NeuroML models
    3. Executes simulation with specified parameters
    4. Generates output files (traces, plots, reports)
    5. Returns execution results

    Expected outputs:
    - Simulation traces in specified format
    - Log files (if configured in LEMS)
    - Performance metrics
    - Error details (if simulation fails)

    Error handling:
    - Invalid LEMS XML will cause syntax errors
    - Missing NeuroML files will cause import errors
    - Simulation runtime errors appear in stderr
    - Check returncode for overall success/failure status

    Performance notes:
    - Large simulations may take significant time
    - Memory usage depends on model complexity and duration
    - Consider limiting simulation scope for testing
    """
    command_args = ["pynml", lems_file]
    request = RunCommand(command=command_args)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)


@tool_meta(ToolInfo(tags={"testing", "neuroml", "neuroml-db"}))
async def get_models_from_neuromldb_tool(
    ctx: Context, search_query: str, num: int = 3, download: bool = False
) -> dict[str, Any]:
    """Use this tool to search and optionally obtain cell and ion channel
    models from the NeuroML model database, NeuroML-DB.

    Common use cases:
    - Finding example cell and channel models
    - Downloading cell and channel models for use

    Args:
        search_query: search term for querying NeuroML-DB. Must be non-empty.
        num: number of search results to get (clamped to 1-20).
        download: set to true to also download the models

    Returns:
        Dictionary of model information with metadata and model content

    Examples:
        # Find and download cerebellar models
        cerebellar_models = get_models_from_neuromldb(search_query="cerebellum", download=True)
    """
    if not search_query or not search_query.strip():
        return {"Error": "search_query must be a non-empty string"}

    num = max(1, min(num, MAX_RESULTS))

    session: aiohttp.ClientSession = ctx.lifespan_context["neuromldb_session"]
    logger.debug(f"{session = }")

    if session is None:
        return {"Error": "NeuroML-DB session not initialized"}

    neuromldb_search_url = "http://neuroml-db.org/api/search"
    neuromldb_model_xml_url = "https://neuroml-db.org/render_xml_file"
    models: dict[str, Any] = {}

    logger.debug(f"Searching NeuroML-DB with query: {search_query}")

    if search_query in NEUROMLDB_SEARCH_CACHE:
        logger.debug(f"Cache hit for search: {search_query}")
        res = NEUROMLDB_SEARCH_CACHE[search_query]
    else:
        try:
            res = await _search_neuromldb(session, neuromldb_search_url, search_query)
            NEUROMLDB_SEARCH_CACHE[search_query] = res
        except Exception as e:
            error_text = f"Error searching NeuroML-DB: {e.__class__.__name__}: {e}"
            logger.error(error_text)
            return {"Error": error_text}

    # Process up to num results
    for i, m in enumerate(res[:num]):
        mcopy = m.copy()

        if download:
            # Rate limit: sleep between requests (but not before the first)
            if i > 0:
                await asyncio.sleep(1)

            model_id = m.get("Model_ID", f"unknown_{i}")

            if model_id in NEUROMLDB_XML_CACHE:
                logger.debug(f"Cache hit for XML: {model_id}")
                mcopy["resource"] = NEUROMLDB_XML_CACHE[model_id]
            else:
                try:
                    xml_path = await _download_file_to_cache_by_content(
                        session,
                        neuromldb_model_xml_url,
                        params={"modelID": model_id},
                        timeout=XML_DOWNLOAD_TIMEOUT,
                        disk_file_name=f"{model_id}.xml",
                    )
                    if xml_path is not None:
                        NEUROMLDB_XML_CACHE[model_id] = xml_path
                        mcopy["resource"] = xml_path
                    else:
                        logger.error(f"Could not get model xml for {model_id}")
                        mcopy["resource"] = ""
                except Exception as e:
                    logger.error(f"Error downloading xml for {model_id}: {e}")
                    mcopy["resource"] = None
        else:
            mcopy["resource"] = None
        models[m.get("Model_ID", f"unknown_{i}")] = mcopy

    return models


@tool_meta(ToolInfo(tags={"testing", "neuroml", "neuroml-db"}))
async def get_repositories_from_open_source_brain_tool(
    ctx: Context,
    search_query: str,
    search_data: bool = True,
    search_models: bool = True,
    num: int = 5,
) -> dict[str, Any]:
    """Use this tool to search the Open Source Brain (v2) neuroscience platform
    (https://v2.opensourcebrain.org) for model and data sources.

    Open Source Brain repositories are an index of a curated set of data and
    code from selected archival platforms (like GitHub, DANDI Archive,
    FigShare). These repositories contain metadata about the data and models,
    including the URL to the storage location. The URL can be passed to other
    tools to download the files.

    Common use cases:
    - Finding neuroscience data and model repositories

    Args:
        search_query: search term for querying Open Source Brain. Must be non-empty.
        search_data: true if data related repositories should be searched
        search_models: true if modelling related repositories should be searched
        num: number of search results to get (clamped to 1-20).

    Returns:
        Dictionary of repository information

    Examples:
        # Find cerebellar models on Open Source Brain
        cerebellar_models = get_repositories_from_open_source_brain(search_query="cerebellum", search_models=True, search_data=False)
        # Find mouse data on Open Source Brain
        mouse_data = get_repositories_from_open_source_brain(search_query="mouse", search_data=True, search_models=False)
        # Find data and models related to the cortex on Open Source Brain
        cortical_repositories = get_repositories_from_open_source_brain(search_query="cortex", search_data=True, search_models=True)
    """
    if not search_query or not search_query.strip():
        return {"Error": "search_query must be a non-empty string"}

    num = max(1, min(num, MAX_RESULTS))

    session: aiohttp.ClientSession = ctx.lifespan_context["osbv2_session"]
    if session is None:
        return {"Error": "OSB session not initialized"}

    # swagger-dev: https://workspaces.v2dev.opensourcebrain.org/api/ui/
    # swagger: https://workspaces.v2.opensourcebrain.org/api/ui/
    osb_repo_search_url = (
        "https://v2.opensourcebrain.org/proxy/workspaces/api/osbrepository"
    )

    # OSB Admin user id: we limit the search to repositories added by the Admin only
    user_id = "7aafb661-2f39-4683-8f35-528de0752dd7"
    query = f"name={search_query}+summary__like=%{search_query}%"

    if search_data and search_models:
        content_types = "experimental+modeling"
    elif search_data:
        content_types = "experimental"
    else:
        content_types = "modeling"

    repositories: dict[str, Any] = {}

    logger.debug(f"Searching OSBv2 repositories with query: {query}")

    cache_key = f"{search_query}+{content_types}"

    if search_query in OSBv2_SEARCH_CACHE:
        logger.debug(f"Cache hit for search: {cache_key}")
        res = OSBv2_SEARCH_CACHE[cache_key]
    else:
        try:
            res = await _search_osbv2_repos(
                session,
                osb_repo_search_url,
                query,
                content_types,
                user_id,
                max_num=num,
            )
            OSBv2_SEARCH_CACHE[cache_key] = res
        except Exception as e:
            error_text = f"Error searching OSBv2: {e.__class__.__name__}: {e}"
            logger.error(error_text)
            return {"Error": error_text}

    # Process up to num results
    logger.debug(f"{res =}")
    results = res["osbrepositories"]

    for i, m in enumerate(results[:num]):
        mcopy = m.copy()

        # remove user information
        del mcopy["user"]

        repositories[m.get("id", f"unknown_{i}")] = mcopy

    return repositories
