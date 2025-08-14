"""
CLI modules for the evaluation pipeline.
"""

# Lazy import to avoid circular import warning
def get_cli():
    """Get the CLI command group."""
    from .main import cli
    return cli

# Only import when explicitly requested
__all__ = [
    "get_cli"
]
