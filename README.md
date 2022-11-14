# AI.Models.CASPR

A deep learning CASPR model.

The following instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

- Install the [x64 version of Python 3.7](https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe).

By default, this will automatically install `pip`. During the installation, check the box to add Python to your PATH environment variable for convenient pip commands.

## Installing

This package can be installed for a developer or for a consumer. You must first get permissions to access the [Business360](https://powerbi.visualstudio.com/Business360/_packaging?_a=feed&feed=Business360) artifact feed to install Python packages.

```bash
pip install twine keyring artifacts-keyring
```

* Installing for a Developer

   Clone the repository to your local machine, and navigate to the root of the repository.

   ```bash
   pip install -r requirements.txt --index-url=https://powerbi.pkgs.visualstudio.com/_packaging/Business360/pypi/simple
   ```

* Installing for a Consumer

   Install the model package from the [Business360](https://powerbi.visualstudio.com/Business360/_packaging?_a=feed&feed=Business360) artifact feed.

   ```bash
   pip install AI.Models.CASPR --index-url=https://powerbi.pkgs.visualstudio.com/_packaging/Business360/pypi/simple
   ```

### Model Versioning
The packages are built from both the release as well as master branches. 
Release branches are versioned from 1.x
Master branch packages are versioned from 0.0.xxx

### Known Issues

1. If you get permission issues, run the same command appended with the `--user`
2. If pip cannot be loaded, prepend the command with `python - m`
3. If it can't find a version that satisfies the requirement, make sure `twine keyring artifacts-keyring` packages are installed and try again with `--user`
4. If no matching distributions are found for the dependency, make sure you have [Python 3.7 x64](https://www.python.org/ftp/python/3.7.0/python-3.7.0-amd64.exe) installed and at the top of your PATH environment variable list

## Linting

We use `pylama` for linting python code, which can be run locally.

1. Run `pip install isort mccabe pylama`
2. At the root of your python project, run `pylama . --options setup.cfg`

VSCode can also be configured to run pylama automatically.

1. Install the Python extension in VSCode
2. Ctrl+Shft+P -> `Python: Enable Linting` -> `on`
3. Ctrl+Shft+P -> `Python: Select Linter` -> `pylama`
4. Ctrl+Shft+P -> `Python: Run Linting`

## Unit Testing

The unit tests were written with the `unittest` library and can be run with `unittest` or `pytest`.

1. Run `pip install unittest`
2. At the root of your python project, run `unittest .`

VSCode can also be configured to detect and run unit tests.

1. Install the Python extension in VSCode
2. Ctrl+Shft+P -> `Python: Configure Tests` -> `unittest` -> `. Root directory` -> `*test*.py`
3. Ctrl+Shft+P -> `Python: Discover Tests`
4. Ctrl+Shft+P -> `Python: Run All Tests`

---

## Copyright (c) Microsoft Corporation. All rights reserved.