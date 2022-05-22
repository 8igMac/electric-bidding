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

## Idea
Use LSTM for predition. Given last 7 days of electricity consumption and
generation, prediction the next 24 hours of difference between consumption
and generation (generation - consumption per hour).

Because trading in the platform is aways better then buying from the market.
I trade when I need to (on every hour). If I have a electricity shortage, I 
buy the missing amount from the bidding platform. If I have more power than 
needed, I always sell it.

For the buying price, I set it to be a little bit lower than the market price, and
let other people get the better price for me.

For the selling price, I set it to be a little bit higher than $0, and let
other get a better price for me.
