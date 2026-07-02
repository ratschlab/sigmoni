"""Thin shim — logic lives in sigmoni.cli. Use 'sigmoni-classify' after pip install."""
from sigmoni.cli import classify

if __name__ == '__main__':
    classify()
