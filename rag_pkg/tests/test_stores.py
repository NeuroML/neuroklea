#!/usr/bin/env python3
"""
Test vector store related code.

File: tests/test_stores.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
import unittest

import pytest
from ollama import ResponseError

from gen_rag.stores import Vector_Stores


class TestStores(unittest.TestCase):
    """Docstring for TestStores."""

    def test_retrieval(self):
        """Test retrieval"""
        try:
            vs_config_file = os.environ.get("GEN_RAG_VS_CONFIG", None)
            stores = Vector_Stores(vs_config_file=vs_config_file)
            stores.setup()
            stores.retrieve("NeuroML", "NeuroML community")
        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))


if __name__ == "__main__":
    unittest.main()
