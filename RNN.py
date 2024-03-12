import torch
import torch.nn as nn
import os
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
import soundfile as sf



class Audio(Dataset):
    def __init__(self, data_dir, target):
        self.extension = ['-input.wav', '-target.wav']
        self.data = []
        for e in self.extension:
            self.file_path = os.path.join(data_dir, target+e)
            audio, self.sr = librosa.load(self.file_path, sr=441000)

            data = []
            # clip audio into half-second segment
            self.seg_len = int(self.sr * 0.5)
            for id in range(0, len(audio), int(self.seg_len)):
                data.append(audio[id:id+self.seg_len])
            data.pop(len(data)-1)

            data = np.array(data)
            data = torch.tensor(data)

            self.len = len(data)
            self.data.append(data)
    
    def __len__(self) -> int:
        return self.len

    def __getitem__(self, index) -> list:
        return self.data[0][index], self.data[1][index]

    def show(self):
        print(f'The audio length is {self.len*0.5} sec')
        print(f'Total {self.len} datapoints')

class RNN(nn.Module):
    def __init__(self, input_size, num_layer, hidden_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layer, batch_first=True)  
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layer, batch_size, self.hidden_size).to(device)
        logic, (h0, c0)= self.rnn(x, (h0, c0))
        out = self.fc(logic)

        return out

def train_one_epoch(model, epoch, train_dataloader, loss_fn, optimiser):
    model.train(True)
    total_loss = 0.0
    for x, y in tqdm(train_dataloader, desc=f'Epoch {epoch+1}', unit='batch'):
        optimiser.zero_grad()
        out = model(x.view(x.size(0), x.size(1), -1))
        loss = loss_fn(out[:, :, 0], y)
        loss.backward()
        optimiser.step()
        total_loss = total_loss+loss
    
    print(f'Total loss: {total_loss}')


if __name__=='__main__':
    hidden_size = 24
    target = 'ht1'
    data_dir = 'Data/train'
    train_input = Audio(data_dir, target)
    train_input.show()
    train_dataloader = DataLoader(train_input, batch_size=40, shuffle=True)

    Train = False

    if torch.cuda.is_available():
        device = 'cuda' 
    else:
        device = 'cpu' 

    seq_len = int(train_input.sr * 0.5)
    model = RNN(input_size=1, hidden_size=hidden_size, num_layer=1)
    model.to(device)

    num_epoch = 10
    learning_rate = 1e-4
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if(Train):
        for epoch in range(num_epoch): 
            train_one_epoch(model, epoch, train_dataloader, loss_fn, optimiser)

    pth_file = 'RNN_model_highpass_prefilt_CrossEntrophy'
    model.load_state_dict(torch.load(pth_file+'_'+str(hidden_size)+'.pth', map_location=torch.device('cpu')))

    input_set = []
    data_dir = 'Data/test'
    input, sr = librosa.load(os.path.join(data_dir, target+'-input.wav'))
    input = torch.tensor(np.array(input)).to(device)
    seg_len = int(sr * 0.5)
    for id in range(0, len(input), int(seg_len)):
        input_set.append(input[id:id+seg_len])
    input_set.pop(len(input_set)-1)
    #print(input_set)
    audio_out = []

    with torch.no_grad():
        for input in input_set:
            #print(input)
            out = model(input.view(input.size(0), 1, -1))[:, 0, 0]
            print(out)
            out = out.numpy()
            audio_out.append(out)

    audio_out = np.concatenate(audio_out, axis=0)
    print(audio_out)
    sf.write(pth_file+'-'+target+'-'+str(hidden_size)+'.wav',audio_out, sr)





        


        

    

