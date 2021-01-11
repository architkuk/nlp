from torch import optim, cuda
from vqa import VQA
from initialize import wordvecDict, parse_questions


def train(questions, images, model, EPOCHS=8):
    optimizer = optim.Adamax(model.parameters(), lr=1e-3)
    for epoch in range(EPOCHS):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model()
        loss = None
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    print("Initializing...")
    glove, glove_path = wordvecDict()
    questions, answers = parse_questions()
    print("Initialized word vecs and qa tensors")
    d = ''
    if cuda.is_available():
        d = 'gpu'
    else:
        d = 'cpu'
    print("Beginning training on ", d)
    train(questions, images, VQA)

