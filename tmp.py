import torch
import random
import sentencepiece as spm
from transformers import ReformerConfig, ReformerModelWithLMHead, ReformerTokenizer
from torch.utils.data import DataLoader, Dataset

NUM_BATCHES = None
BATCH_SIZE = 12
GRADIENT_ACCUMULATE_EVERY = 6
LEARNING_RATE = 0.01
VALIDATE_EVERY  = 20
SEQ_LEN = 24576


spm.SentencePieceTrainer.Train("--input=./data/tokenizer_training/AAresiduals.txt \
                                --vocab_size=28 \
                                --model_prefix=sequence_tokenizer \
                                --model_type=char \
                                --character_coverage=1.0")
tokenizer = ReformerTokenizer(vocab_file="sequence_tokenizer.model", do_lower_case=False, model_max_length=SEQ_LEN)


# configuration = ReformerConfig.from_pretrained("google/reformer-crime-and-punishment")
# configuration.axial_pos_shape=(128, 192)
# configuration.max_position_embeddings=SEQ_LEN
# configuration.vocab_size=tokenizer.vocab_size
# configuration.save_pretrained('model/config/')

configuration = ReformerConfig.from_pretrained('model/config/')
model = ReformerModelWithLMHead(configuration)

def split_file(file,out1,out2,percentage=0.75,isShuffle=True,seed=42):
    """quora.com/How-can-split-a-text-file-randomly-in-75-and-25-and-create-two-output-file-in-python
    """
    random.seed(seed)
    with open(file, 'r',encoding="utf-8") as fin, open(out1, 'w') as foutBig, open(out2, 'w') as foutSmall:
        nLines = sum(1 for line in fin)
        fin.seek(0)

        nTrain = int(nLines*percentage) 
        nValid = nLines - nTrain

        i = 0
        for line in fin:
            r = random.random() if isShuffle else 0 # so that always evaluated to true when not isShuffle
            if (i < nTrain and r < percentage) or (nLines - i > nValid):
                foutBig.write(line)
                i += 1
            else:
                foutSmall.write(line)
                
split_file("data/yeast/yeast.txt", 
           "data/yeast/yeast_train.txt",
           "data/yeast/yeast_val.txt",
           percentage=0.9)

def cycle(loader):
    while True:
        for data in loader:
            yield data


class SequenceDataset(Dataset):
    def __init__(self, inputs, tokenizer, _len):
        super().__init__()
        self.inputs = inputs
        self.tokenizer = tokenizer
        self._len = _len

    @classmethod
    def prepare_from_file(cls, file_path, tokenizer):
        with open(file_path) as file:
            X = [l.strip() for l in file]
            X = [tokenizer.encode(sequence, 
                                  max_length=tokenizer.max_len, 
                                  add_special_tokens=True, 
                                  pad_to_max_length=True) for sequence in X]
            X = [torch.tensor(sequence) for sequence in X]
        inputs = torch.stack([X[i] for i in range(len(X))]).squeeze()
        return cls(inputs, tokenizer, len(inputs))

    def __getitem__(self, index):
        return self.inputs[index, ].cuda()

    def __len__(self):
        return self._len

train_dataset = SequenceDataset.prepare_from_file("data/yeast/yeast_train.txt", tokenizer)
val_dataset = SequenceDataset.prepare_from_file("data/yeast/yeast_val.txt", tokenizer)
train_loader = cycle(DataLoader(train_dataset, batch_size=BATCH_SIZE))
val_loader = cycle(DataLoader(val_dataset, batch_size=BATCH_SIZE))


from transformers import AdamW
from torch.nn import CrossEntropyLoss

optimizer = AdamW(params=model.parameters(), lr=LEARNING_RATE)

NUM_BATCHES = len(train_dataset)//BATCH_SIZE

model.cuda()

import tqdm

for x in range(3):

    for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
        
        model.train()
        
        for __ in range(GRADIENT_ACCUMULATE_EVERY):
            inputs = next(train_loader)
            outputs = model(inputs, labels=inputs)
            loss, prediction_scores = outputs[:2]
            loss.backward()

        print(f'training loss: {loss.item()}')
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        optimizer.step()
        optimizer.zero_grad()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            with torch.no_grad():
                inputs = next(val_loader)
                outputs = model(inputs, labels=inputs)
                loss, prediction_scores = outputs[:2]
                print(f'validation loss: {loss.item()}')
                
    torch.save(model.state_dict(), f'model/trial01/saved_model_epoch_{x}.pth')




