"""
Backward-compatible wrapper for `scripts/rsa/run_rsa_matrix.py`.

The canonical RSA matrix runner lives under `scripts/rsa/`.
"""

from scripts.rsa.run_rsa_matrix import *  # noqa: F401,F403


if __name__ == "__main__":
    from scripts.rsa.run_rsa_matrix import main

    main()


