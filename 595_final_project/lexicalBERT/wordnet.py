import json
import torch

from torch import nn
from torch import cuda
from torch.optim import Adam

from torchkge.models import TransEModel
from torchkge.models.translation import TransHModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
from torchkge.utils.datasets import load_fb15k, load_wn18rr

from tqdm.autonotebook import tqdm

# Load dataset
kg = load_wn18rr()

# Define some hyper-parameters for training
emb_dim = 300
lr = 0.0004
n_epochs = 1000
b_size = 32768
margin = 0.5


class WordNetEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.kg = load_wn18rr()
        self.embedding = nn.Embedding(num_embeddings=kg.n_ent, embedding_dim=emb_dim)
        self.embedding.weight.data = self.init_embedding()

        self.transform = nn.Sequential(
            nn.Linear(emb_dim, 768)
        )

        with open('bert2wn.json', 'r') as file:
            self.bert2wn = json.load(file)

    def forward(self, input_ids, input_tensors):
        wordnet_emb = []
        shape1, shape2 = input_ids.size()
        for id in input_ids.view(-1):
            key = str(int(id))
            value = self.bert2wn[key]
            if len(value) == 0:
                wordnet_emb.append(torch.zeros(emb_dim).cuda())
            else:
                wordnet_emb.append(torch.mean(self.embedding(torch.tensor(value).cuda()), dim=0))

        wordnet_emb = torch.stack(wordnet_emb).view(shape1, shape2, emb_dim) # [32, 128, 100]
        # wordnet_bert_emb = torch.cat([wordnet_emb, input_tensors], dim=2)

        return self.transform(wordnet_emb) + input_tensors

    def init_embedding(self):
        model = TransEModel(emb_dim, kg.n_ent, kg.n_rel)
        criterion = MarginLoss(margin)
        if cuda.is_available():
            cuda.empty_cache()
            model.cuda()
            criterion.cuda()

        model.train()
        # Define the torch optimizer to be used
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        sampler = BernoulliNegativeSampler(kg)
        dataloader = DataLoader(kg, batch_size=b_size, use_cuda='all')

        iterator = tqdm(range(n_epochs), unit='epoch')
        for epoch in iterator:
            running_loss = 0.0
            for i, batch in enumerate(dataloader):
                h, t, r = batch[0], batch[1], batch[2]
                n_h, n_t = sampler.corrupt_batch(h, t, r)

                optimizer.zero_grad()

                # forward + backward + optimize
                pos, neg = model(h, t, n_h, n_t, r)
                loss = criterion(pos, neg)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            iterator.set_description(
                'Epoch {} | mean loss: {:.5f}'.format(epoch + 1,
                                                    running_loss / len(dataloader)))

        model.normalize_parameters()
        ent_embedding, _ = model.get_embeddings()

        return ent_embedding