import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        
        
        self.word_emb = nn.Embedding(vocab_size, embed_size)
        
        print(self.word_emb.weight.type())
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, self.num_layers, 
                            batch_first=True)
        
        # self.dropout = nn.Dropout(drop_prob)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        # input entire sequence as suggested in the project introduction
        
        
        embeddings = self.word_emb(captions)  # (batch_size, max_caption_length, embed_dim)
        
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        out, hidden = self.lstm(embeddings)
        
        out_captions = self.fc(out[:, :-1, :]) # we don't need the last output which takes the input <end>
        return out_captions
       
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass