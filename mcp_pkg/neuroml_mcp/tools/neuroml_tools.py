#!/usr/bin/env python3
"""
Tools for generating NeuroML code

File: codegen_tools.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict

from neuroml_mcp.tools.sandbox.sandbox import RunCommand

# set the implementation for development
from ..utils import ToolInfo, tool_meta
from .sandbox import nml_mcp_sandbox

sbox = nml_mcp_sandbox


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
