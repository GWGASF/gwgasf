Copy paste the following command to create the enviroment to run GASF-data-generation-example.

```
conda create -n GASF python==3.9 -y
conda activate GASF
pip install poetry==1.7.1

cd ml4gw
poetry install
cd ../libs/gasf
poetry install

```
