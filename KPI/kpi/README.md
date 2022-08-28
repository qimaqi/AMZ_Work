# AMZ KPI Toolsuite

## Usage

Quick help and usage instructions are available using `--help`.


## Installation

Follow the installation instructions for the main repository.
Then, do:

```bash
# Install the AMZ KPI Toolsuite
pip3 install -e tools/kpi/.

# Test installation
amz_kpi --help

# if you run into " ModuleNotFoundError: No module named 'src.dummy_kpi' "
# add the path to the kpi folder to PYTHONPATH as a workaround
export PYTHONPATH="$PYTHONPATH:<path_to_kpi_directory>"
```
