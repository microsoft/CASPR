from setuptools import setup

# replaced by AI.Common build template
auto_replaced = "__version__"

# minor trick to circumvent version warning when building manually
version = None if 'version' in auto_replaced else auto_replaced

setup(version=version)
