"""Thin shim — logic lives in sigmoni.cli. Use 'sigmoni-index' after pip install."""
from sigmoni.cli import index

if __name__ == '__main__':
    index()
