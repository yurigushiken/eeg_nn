"""
Backward-compatible wrapper for `scripts/rsa/compile_rsa_results.py`.
"""

from scripts.rsa.compile_rsa_results import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.rsa.compile_rsa_results import main

    main()


