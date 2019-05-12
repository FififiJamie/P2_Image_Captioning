import torch
import torch.nn as nn
import torchvision.models as models

import torch.nn.functional as F

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
        
        # features is  (batch_size, embed_dim)
        # captions is (batch_size, max_caption_length)
        
        embeddings = self.word_emb(captions)  # (batch_size, max_caption_length, embed_dim)
        
        # turn feature into (batch_size, 1, embed_dim) then append embedings
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        out, hidden = self.lstm(embeddings) # (batch_size, max_caption_length + 1, hidden_size)
        
        # we don't need the last output which takes '<end>' as input
        out_captions = self.fc(out[:, :-1, :])  # (batch_size, max_caption_length, vocab_size)
        
        return out_captions
    
    # Ugly beam search ...
    def beam_search(self, beam_width, max_words, in_p, h, c):
        
        # ini first nodes for search
        hiddens = (h, c)
        for _ in range(beam_width - 1):
            hiddens = (torch.cat((hiddens[0], h), 1), torch.cat((hiddens[1], c), 1)) 
            
        ps, top_words = in_p.topk(beam_width) 
        h = hiddens[0]
        c = hiddens[1]
        
        # List of strings containing search results
        sentences = []
        for i in range(beam_width):
            sentences.append([top_words.view(-1, 1)[i].item()])
        
        # Search 
        for _ in range(max_words):
            # Put search in a batch (batch size = beam_width)
            outs = self.word_emb(top_words.view(-1, 1)) 
            outs, hiddens = self.lstm(outs, (h,c)) # (beam_width, 1, vocab_size)
            out_captions = F.softmax(self.fc(outs), 2)
            
            # Get top k prdictions. ps_next = probabilities, top_words_next = words
            # Say beam_width is 3, top_words_next will look like this:
            #   [[[Words_11, Words_12, Words_13]]
            #    [[Words_21, Words_22, Words_23]]
            #    [[Words_31, Words_32, Words_33]]]
            # Where row id corresponds to previous nodes
            ps_next, top_words_next = out_captions.topk(beam_width) # (beam_width, 1, beam_width)
            print(top_words_next.shape)

            # Update conditional probabilities
            new_ps = ps_next * ps.view(beam_width, -1, 1) 
            
            # Get top k results. Note that top_words_indexes_flat is the flatten index pointing to top_words_next
            ps, top_words_indexes_flat = new_ps.view(-1).topk(beam_width)

            # Match back to sentences. 
            
            x = top_words_indexes_flat / beam_width # Row id corresponds to previous words

            new_sentences = []

            # Build input for next search
            next_words = []
            next_h = []
            next_c = []
            for i in range(beam_width):
                next_words.append((top_words_next.view(-1))[top_words_indexes_flat[i]])
                next_h.append(hiddens[0][0][x[i]])
                next_c.append(hiddens[1][0][x[i]])

            next_words = torch.stack(next_words)
            next_h = torch.stack(next_h).unsqueeze(0)
            next_c = torch.stack(next_c).unsqueeze(0)

            # Get strings so far
            for i in range(beam_width):
                a_sentence = []
                a_sentence = sentences[x[i]] + [(top_words_next.view(-1))[top_words_indexes_flat[i]].item()]
                new_sentences.append(a_sentence)

            # Prepare for next iteration
            top_words = next_words
            acc_p = ps
            h = next_h
            c = next_c
            sentences = new_sentences
            
        return new_sentences

        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        caption_tokens = []
        
        
        # first word
        out, hidden = self.lstm(inputs) 
        out_captions = F.softmax(self.fc(out), 2) # we might not need softmax here since we are choosing the max
        _, out_indexes = torch.max(out_captions, 2)
        
        caption_tokens.append(out_indexes[0].item())
        
        
        # Beam search, ini first nodes
#         beam_width = 3
#         hiddens = hidden
#         for _ in range(beam_width - 1):
#             hiddens = (torch.cat((hiddens[0], hidden[0]), 1), torch.cat((hiddens[1], hidden[1]), 1)) 
            
#         ps, top_words = out_captions.topk(beam_width) 
        
        beam_width = 3
        beam_sentences = self.beam_search(beam_width, max_len, out_captions, hidden[0], hidden[1])
      
        for i in range(max_len - 1):
            out = self.word_emb(out_indexes[0].unsqueeze(0))
            
            out, hidden = self.lstm(out, hidden)
            
            out_captions = F.softmax(self.fc(out), 2) # we might not need softmax here since we are choosing the max
            
            _, out_indexes = torch.max(out_captions, 2)
            
            caption_tokens.append(out_indexes[0].item())
            
            
            
        return caption_tokens, beam_sentences;