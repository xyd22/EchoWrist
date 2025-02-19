import torch.nn as nn

# Recurrent neural network (many-to-one)
class myRNN(nn.Module):
    def __init__(self, rnn_config):
        super(myRNN, self).__init__()
        # self.device = device
        self.hidden_size = rnn_config['hidden_size']
        self.num_layers = rnn_config['num_layers']
        self.bidirectional = rnn_config['bidirectional']

        self.rnn_type = rnn_config['rnn_type'].lower()
        if self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                rnn_config['input_size'],
                rnn_config['hidden_size'],
                rnn_config['num_layers'],
                batch_first=True,
                bidirectional=rnn_config['bidirectional']
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                rnn_config['input_size'],
                rnn_config['hidden_size'],
                rnn_config['num_layers'],
                batch_first=True,
                bidirectional=rnn_config['bidirectional']
            )
        # self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x, state=None):
        # print(x.shape)
        # Set initial hidden and cell states 
        # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.device)
        # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).cuda(self.device)
        # Forward propagate LSTM
        out, _ = self.rnn(x, state)  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        # out = self.fc(out[:, -1, :])
        return out

def rnn(rnn_config, **kwargs):
    model = myRNN(rnn_config)
    return model