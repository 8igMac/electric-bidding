"""
Define the model.
"""
import torch.nn as nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        """ hidden_size: for LSTM.
        """
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, days):
        out, _ = self.lstm(days)  # (batch_size, 7 days, 24 hours)
        out = self.fc(out[:,-1:,:])
        return out
