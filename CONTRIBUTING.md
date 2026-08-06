# Contributing

Thank you for your interest in the project. This repository is primarily a reproducible academic artifact, but focused corrections and reproducibility improvements are welcome.

## Before opening a change

1. Create a branch from `main`.
2. Use Python 3.11 and a fresh virtual environment.
3. Install the lightweight test dependencies with `python -m pip install -r requirements-test.txt`.
4. Keep generated model weights, dense embeddings, credentials, and local experiment logs out of Git.
5. Explain whether a result is newly computed or copied from an existing committed artifact.

## Checks

Run these commands from the repository root:

```bash
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

If a notebook changes, ensure that it is valid JSON, clear machine-specific paths, and state the environment used to produce new outputs. Avoid committing large binary cell outputs unless they materially support the analysis.

## Pull requests

A useful pull request should include:

- a concise problem statement;
- the method or files changed;
- evidence that the checks pass;
- before/after metrics for modelling changes;
- provenance and license information for any new dataset, model, or figure.

The repository's original material is not currently distributed under an open-source license. A contribution does not imply permission to reuse unrelated material from the project.
