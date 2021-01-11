import torch
import torch.nn as nn
import torch.nn.functional as F


class VQA(nn.Module):
    def __init__(self, glove_path, input_size, hidden_size, num_classes=3129):
        self.qemb = QuestionEmbedding(input_size, glove_path)
        self.att = ImageAttention(hidden_size, 2048)
        self.classifier = Classifier(hidden_size, num_classes)
        self.lin1 = nn.Linear(hidden_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.nlin1 = NonLinear(hidden_size)
        self.nlin2 = NonLinear(hidden_size)

    # images should be K x 2048
    def forward(self, question, image):
        q = self.qemb(question)
        vHat = self.att(image)
        h = torch.mul(self.nlin1(q), self.nlin2(vHat))
        sHat = self.classifier(h)
        return sHat


class QuestionEmbedding(nn.Module):
    def __init__(self, input_size, glove_path):
        self.emb = nn.Embedding(input_size+1, 300)
        self.gru = nn.GRU(300, 512) # 512
        # load pretrained model weights

    # x => batch x 14 x 300
    def forward(self, x):
        emb = self.emb(x)
        output, _ = self.gru(emb)
        return output


class ImageAttention(nn.Module):
    def __init__(self, input_size):
        self.lin = nn.Linear(input_size, input_size)
        self.f = NonLinear(input_size)

    # input should be v (image features), q (question embeddings)
    # in step 1 (lin(f(x))), x => concatenate(v, q)
    def forward(self, v, q):
        x = torch.concat(v, q)
        a = self.lin(self.f(x))
        a = F.softmax(a)
        return x


class NonLinear(nn.Module):
    def __init__(self, input_size):
        self.lin1 = nn.Linear(input_size, input_size, bias=True)
        self.lin2 = nn.Linear(input_size, input_size, bias=True)

    def forward(self, x):
        yTilde = F.tanh(self.lin1(x))
        g = F.sigmoid(self.lin2(x))
        return torch.mul(yTilde, g)


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        self.lin = nn.Linear(input_size, num_classes, bias=False)
        self.f = NonLinear(input_size)

    def forward(self, x):
        x = self.lin(self.f(x))
        x = F.sigmoid(x)    # N x 512
        return x
