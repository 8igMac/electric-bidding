# Electric Bidding

## Usage
- Install Dependencies
```sh
$ pipenv install
```
- Train the model. (The model will be stored as `{model_name}.hdf5`)
```sh
$ pipenv run python train.py
```
- Generate biding plan.
```sh
$ pipenv run python main.py \
    --consumption ./sample_data/consumption.csv \
    --generation ./sample_data/generation.csv \
    --bidresult ./sample_data/bidresult.csv 
    --output output.csv
```
