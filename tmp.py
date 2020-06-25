import torch
import random
import time
import numpy as np
import sentencepiece as spm
from transformers import ReformerConfig, ReformerModelWithLMHead, ReformerTokenizer, EncoderDecoderConfig, EncoderDecoderModel
from torch.utils.data import DataLoader, Dataset

NUM_BATCHES = None
BATCH_SIZE = 20
LEARNING_RATE = 0.001 #1e-4 #1e-4
VALIDATE_EVERY  = 10
SEQ_LEN = 4608

# spm.SentencePieceTrainer.Train("--input=./data/tokenizer_training/AAresiduals.txt \
#                                 --vocab_size=28 \
#                                 --model_prefix=sequence_tokenizer \
#                                 --model_type=char \
#                                 --character_coverage=1.0")
tokenizer = ReformerTokenizer(vocab_file="sequence_tokenizer.model", do_lower_case=False, model_max_length=SEQ_LEN)
tokenizer.max_model_input_sizes = SEQ_LEN

# def split_file(file,out1,out2,percentage=0.75,isShuffle=True,seed=42):
#     random.seed(seed)
#     with open(file, 'r',encoding="utf-8") as fin, open(out1, 'w') as foutBig, open(out2, 'w') as foutSmall:
#         nLines = sum(1 for line in fin)
#         fin.seek(0)

#         nTrain = int(nLines*percentage) 
#         nValid = nLines - nTrain

#         i = 0
#         for line in fin:
#             r = random.random() if isShuffle else 0 # so that always evaluated to true when not isShuffle
#             if (i < nTrain and r < percentage) or (nLines - i > nValid):
#                 foutBig.write(line)
#                 i += 1
#             else:
#                 foutSmall.write(line)
                
# split_file("data/yeast/yeast.txt", 
#            "data/yeast/yeast_train.txt",
#            "data/yeast/yeast_val.txt",
#            percentage=0.9)

class SequenceDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels, tokenizer, _len):
        super().__init__()
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels
        self.tokenizer = tokenizer
        self._len = _len

    @classmethod
    def prepare_from_file(cls, file_path, tokenizer):
        
        with open(file_path) as file:
            
            X = [l.strip() for l in file]
            X = [tokenizer.encode(sequence)[1:tokenizer.max_len+1] for sequence in X]
            
            temp = [tokenizer.prepare_for_model(sequence) for sequence in X]
            
            input_ids = [np.pad(x["input_ids"], 
                                (0, tokenizer.max_len - len(x["input_ids"])), 
                                'constant', constant_values=0) for x in temp]

            attention_mask = [np.pad(x["attention_mask"], 
                                     (0, tokenizer.max_len - len(x["attention_mask"])),
                                     'constant', constant_values=0) for x in temp]
            
            labels = [np.pad(x["input_ids"], 
                             (0, tokenizer.max_len - len(x["input_ids"])), 
                             'constant', constant_values=-100) for x in temp]

            input_ids = [torch.tensor(x, dtype=torch.int64) for x in input_ids]
            attention_mask = [torch.tensor(x, dtype=torch.int64) for x in attention_mask]
            labels = [torch.tensor(x, dtype=torch.int64) for x in labels]
            
            input_ids = torch.stack([input_ids[i] for i in range(len(input_ids))]).squeeze()
            attention_mask = torch.stack([attention_mask[i] for i in range(len(attention_mask))]).squeeze()
            labels = torch.stack([labels[i] for i in range(len(labels))]).squeeze()
            
            del(temp); del(X);
        return cls(input_ids, attention_mask, labels, tokenizer, len(input_ids))

    def __getitem__(self, index):
        return {"input_ids": self.input_ids[index, ].cuda(), 
                "attention_mask": self.attention_mask[index, ].cuda(),
                "labels": self.labels[index, ].cuda()}

    def __len__(self):
        return self._len


def cycle(loader):
    while True:
        for data in loader:
            yield data

train_dataset = SequenceDataset.prepare_from_file("data/yeast/yeast_train.txt", tokenizer)
val_dataset = SequenceDataset.prepare_from_file("data/yeast/yeast_val.txt", tokenizer)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))

# configuration = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
# configuration.axial_pos_shape = (64, 72)
# configuration.max_position_embeddings=SEQ_LEN
# configuration.vocab_size=tokenizer.vocab_size
# configuration.save_pretrained('model/config/')
configuration = ReformerConfig.from_pretrained('model/config/')
model = ReformerModelWithLMHead(configuration)
model.cuda()

NUM_BATCHES = len(train_dataset)//BATCH_SIZE

from transformers import AdamW
optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)

from collections import OrderedDict 
import json

all_training_loss = OrderedDict()
all_val_loss = OrderedDict()

for x in range(1):
    print(f"epoch {x}")
    start = time.time()

    training_loss = OrderedDict()
    val_loss = OrderedDict()
    
    for i in range(NUM_BATCHES):
        print("step {}".format(i))
        model.train()

        tmp = next(train_loader)
        input_ids = tmp['input_ids']
        attention_mask = tmp['attention_mask']
        labels = tmp['labels']

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, prediction_scores = outputs[:2]
        loss.backward()
        
        training_loss[f"Epoch {x} Step {i}"] = loss.item()
        all_training_loss[f"Epoch {x} Step {i}"] = loss.item()
        print(f'training loss: {loss.item()}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

        optimizer.step()
        optimizer.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                tmp = next(val_loader)
                input_ids = tmp['input_ids']
                attention_mask = tmp['attention_mask']
                labels = tmp['labels']
                
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss, prediction_scores = outputs[:2]

                val_loss[f"Epoch {x} Step {i}"] = loss.item()
                all_val_loss[f"Epoch {x} Step {i}"] = loss.item()
                print(f'validation loss: {loss.item()}')
                
    torch.save(model.state_dict(), f"saved_0624/model/saved_model_epoch_{x}.pth")
    
    with open(f'saved_0624/saved_losses/training_loss_epoch_{x}.json', 'w') as f:
        f.write(json.dumps(training_loss))
    
    with open(f'saved_0624/saved_losses/val_loss_epoch_{x}.json', 'w') as f:
        f.write(json.dumps(val_loss))
    end = time.time()
    print(f"----------{(end-start)//60} min per epoch----------")

with open("saved_0624/saved_losses/training_loss_all.json", 'w') as f:
    f.write(json.dumps(all_training_loss))
with open("saved_0624/saved_losses/val_loss_all.json", 'w') as f:
    f.write(json.dumps(all_val_loss))


