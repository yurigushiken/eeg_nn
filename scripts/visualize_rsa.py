"""
Backward-compatible wrapper for `scripts/rsa/visualize_rsa.py`.

Some tests and older docs refer to `scripts/visualize_rsa.py`. The canonical
implementation lives under `scripts/rsa/`.
"""

from scripts.rsa.visualize_rsa import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.rsa.visualize_rsa import main

    main()


