# docs/conf.py
import os
import sys
sys.path.insert(0, os.path.abspath(".."))

project = 'Ballistic‑Simulation‑Engine'
author = 'Utkarsh Dwivedi'
release = '1.0.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
]

autodoc_default_flags = ['members', 'undoc-members', 'private-members']
autodoc_member_order = 'bysource'
autoclass_content = 'both'

templates_path = ['_templates']
exclude_patterns = []
html_theme = 'alabaster'
# html_static_path = ['_static']
