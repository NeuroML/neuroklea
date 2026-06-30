Contributing
============

Development setup
-----------------

Clone the repository and install in editable mode::

   git clone https://github.com/NeuroML/neuroklea.git
   cd neuroklea
   pip install -r requirements-dev.txt

Workflow
--------

This project uses a standard fork-and-PR workflow:

1. Fork the repository on GitHub.
2. Create a feature branch from ``development``.
3. Make your changes and run verification (lint, typecheck, test).
4. Submit a pull request targeting the ``development`` branch.
5. Address review feedback if any.

Commands
--------

Lint and format
~~~~~~~~~~~~~~~

.. code-block:: bash

   ruff check . --fix
   ruff format .

Type check
~~~~~~~~~~

.. code-block:: bash

   ty

Run tests
~~~~~~~~~

.. code-block:: bash

   # All tests
   bash scripts/run_tests.sh

   # Single package
   cd utils_pkg && pytest -v

   # Exclude tests that need an LLM
   pytest -m "not localonly"

Pre-commit
~~~~~~~~~~

.. code-block:: bash

   pre-commit run --all-files

CLI verification
~~~~~~~~~~~~~~~~

If you modify a CLI entry point, confirm it starts::

   <cli-name> --help

AI use
------

This repository includes an ``AGENTS.md`` file with instructions for AI
coding assistants.  While it is fine to use AI for development, it is
expected that the human reviews every line of code before submitting PRs.
PRs that are AI only and have not been checked by humans will not be
accepted.  All AI assistance must be clearly noted in the PR
description.

Pull requests
-------------

* Open PRs targeting the ``development`` branch.
* Keep changes focused -- one logical change per PR.
* Ensure CI passes (lint, typecheck, tests).
* Include a conventional commit message with issue numbers when applicable.
