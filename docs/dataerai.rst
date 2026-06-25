Dataerai provenance tutorials
=============================

Pycroscopy notebooks can preserve complete notebook runs in
`Dataerai <https://dataerai.com/>`_ by calling the Dataerai Python SDK directly.
This guide does not add a pycroscopy API or dependency. Install and authenticate
Dataerai only in notebook environments that need to preserve assets.

Installation
------------

.. code:: bash

  pip install dataerai
  dataerai auth login

If your notebook should start the local Dataerai daemon automatically, set the
``binary_path`` argument in ``DataeraiClient`` to the installed ``dataerai``
binary. If the daemon is already running, the default socket connection is
enough.

Notebook-run workflow
---------------------

Every notebook under ``jupyter_notebooks`` includes an opt-in Dataerai section.
The section follows the same pattern as Dataerai's PyTorch training helpers:
SDK calls handle file transfers and provenance relationships, while Dataerai's
credential-refresh helper is used for REST endpoints that are not yet exposed on
``DataeraiClient``.

When enabled, each notebook preservation cell:

1. Creates a notebook-run collection under the selected project, or reuses the
   collection in ``DATAERAI_COLLECTION_ID``.
2. Preserves local raw input files listed in ``DATAERAI_RAW_DATA_PATHS``.
3. Preserves a raw-input manifest with parameters and missing raw-path audit
   details.
4. Preserves the notebook file when it can be found, plus a notebook manifest
   containing environment, git, and IPython input-history information.
5. Preserves derived array artifacts, additional output files, and output
   images, including any currently open matplotlib figures.
6. Preserves an execution log that records the run id, collection id, uploaded
   asset ids, content ids, source asset ids, missing paths, parameter summary,
   and package versions.
7. Creates provenance relationships so outputs are linked to raw inputs,
   notebook assets, execution logs, and output images.

Configuration
-------------

Each notebook cell starts with editable settings:

.. code:: python

  RUN_DATAERAI_DEMO = False
  DATAERAI_PROJECT_ID = "<your-project-id>"
  DATAERAI_BINARY_PATH = None
  DATAERAI_SOURCE_ASSET_IDS = []
  DATAERAI_PARENT_COLLECTION_ID = None
  DATAERAI_COLLECTION_ID = None

Set ``RUN_DATAERAI_DEMO = True`` and replace ``DATAERAI_PROJECT_ID`` before
running the cell. Use ``DATAERAI_SOURCE_ASSET_IDS`` when an upstream raw asset
already exists in Dataerai. Use ``DATAERAI_COLLECTION_ID`` to preserve repeated
runs into the same collection; otherwise the cell creates a new collection for
each run.

Captured metadata
-----------------

Every uploaded asset receives a top-level ``pycroscopy`` metadata object. The
shape is intentionally simple so Dataerai records can be searched and audited
without requiring Dataerai to understand every pycroscopy object:

.. code:: json

  {
    "pycroscopy": {
      "schema_version": 2,
      "run_id": "...",
      "created_at": "2026-06-25T00:00:00+00:00",
      "notebook": "Intro_to_Pycroscopy.ipynb",
      "operation": "intro_pycroscopy_analysis",
      "role": "derived_result",
      "file": {
        "filename": "pycroscopy_intro_result.npz",
        "suffix": ".npz",
        "size_bytes": 32768,
        "sha256": "..."
      },
      "environment": {
        "python": "...",
        "platform": "...",
        "packages": {
          "pycroscopy": "...",
          "dataerai": "..."
        },
        "git": {
          "commit": "...",
          "branch": "...",
          "dirty": false
        }
      },
      "artifact": {
        "dataset_sid": {
          "shape": [64, 64],
          "dtype": "float64"
        }
      },
      "lineage": {
        "source_asset_ids": ["raw-asset-id"]
      }
    }
  }

Relationship types
------------------

The notebook cells create directed relationships from dependent assets to their
upstream evidence:

* ``derived_from`` links result and derived-output assets to raw input assets,
  source assets, or the raw-input manifest.
* ``copy_of`` links uploaded raw local files to existing Dataerai source assets
  when ``DATAERAI_SOURCE_ASSET_IDS`` is provided.
* ``generated_by`` links outputs and images to the notebook manifest or notebook
  snapshot.
* ``documented_by`` links outputs and images to the execution-log asset.
* ``visualization_of`` links output-image assets to the derived-result asset.
* ``execution_of`` links the execution-log asset to the notebook manifest.

File format follow-up
---------------------

The notebook examples save compact NumPy ``.npz`` result bundles and preserve
raw files in their original local formats when those files are present. A
follow-up ADR should decide the preferred preservation formats for pycroscopy
workflows, including when to preserve original source files, ``sidpy``/HDF5
artifacts, compact derived arrays, figures, and execution logs.
