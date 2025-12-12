"""
Backward-compatible wrapper for `scripts/rsa/run_temporal_decoding.py`.
"""

from scripts.rsa.run_temporal_decoding import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.rsa.run_temporal_decoding import main

    main()


