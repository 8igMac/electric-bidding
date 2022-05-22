"""
Load the model and output biding plan for tormorrow.
"""
import params
import torch
import numpy as np
import pandas as pd
from model import MyLSTM
from datetime import datetime, timedelta

# You should not modify this part.
def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()

def training_and_eval():
    # Training.

    # Evaluating.
    pass

def output(path, data):
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)
    return

def load_data(consumption_file, generation_file, bidresult_file):
    con_df = pd.read_csv(consumption_file, parse_dates=True, index_col=0, dtype=np.float32)
    gen_df = pd.read_csv(generation_file, parse_dates=True, index_col=0, dtype=np.float32)
    df = pd.concat([con_df, gen_df], axis=1)
    df['gen-con'] = df['generation'] - df['consumption']
    tensor = torch.from_numpy(df['gen-con'].to_numpy())
    tensor = tensor.reshape(7, 24)

    return tensor

def gen_biding_plan(outputs, target_date):
    date_list = pd.date_range(start=target_date, freq='H', periods=24)

    data = []
    for date, output in zip(date_list, outputs):
        action = 'buy' if output < 0 else 'sell'
        price = 2.51 if action == 'buy' else 2.53
        volumn = abs(round(output, 2))
        data.append([date.strftime('%Y-%m-%d %H:%M:%S'), action, price, volumn])
        
    return data

def get_target_date(consumption_file):
    df = pd.read_csv(consumption_file, parse_dates=True, date_parser='%')
    date_str = df['time'].iloc[-1]
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    target_date = date + timedelta(days=1)
    return target_date.strftime('%Y-%m-%d')

if __name__ == "__main__":
    args = config()

    # Load the model.
    model = MyLSTM(params.INPUT_SIZE, params.OUTPUT_SIZE, params.HIDDEN_SIZE)
    model.load_state_dict(torch.load('./model.hdf5'))
    model.eval()

    # Read the data.
    inputs = load_data(args.consumption, args.generation, args.bidresult)
    assert inputs.shape == (7, 24)

    # Inference.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        inputs = inputs.unsqueeze(0)
        assert inputs.shape == (1, 7, 24)

        inputs = inputs.to(device)
        model = model.to(device)

        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()
        assert outputs.shape == (1, 1, 24)
        outputs = outputs.reshape(-1)
        assert outputs.shape == (24,)

    # Make biding decision.
    target_date = get_target_date(args.consumption)
    data = gen_biding_plan(outputs, target_date)

    # Output bidding result.
    output(args.output, data)
