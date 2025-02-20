# Source Directory

Directory containing the active ***src*** codebase as well as old and unused codebases.

main.py and arguments.toml are the current script control files. All scripts for main.py are contained in ***libs***.



Copy paste the following command to create the enviroment to run GASF-data-generation-example.

```
conda create -n GASF python==3.9 -y
conda activate GASF
pip install poetry==1.7.1

cd ../images/
poetry install

```