#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn
import torch.nn.functional as F

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway


# End "do not change"

class ModelEmbeddings(nn.Module):
    """
    Class that converts input words to their CNN-based embeddings.
    """

    def __init__(self, word_embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param word_embed_size (int): Embedding size (dimensionality) for the output word
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.

        Hints: - You may find len(self.vocab.char2id) useful when create the embedding
        """
        super(ModelEmbeddings, self).__init__()

        ### YOUR CODE HERE for part 1h
        self.vocab = vocab
        self.e_char = 50
        self.e_word = word_embed_size
        self.emb = nn.Embedding(len(self.vocab.char2id), self.e_char, padding_idx=self.vocab.char_pad)
        self.conv = CNN(self.e_char, self.e_word)
        self.highway = Highway(self.e_word)
        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, word_embed_size), containing the
            CNN-based embeddings for each word of the sentences in the batch
        """
        ### YOUR CODE HERE for part 1h
        x_emb = self.emb(input)
        slen, bsize, eword, echar = x_emb.shape
        x_reshaped = x_emb.reshape(slen*bsize, eword, echar).transpose(1, 2)
        x_conv_out = self.conv(x_reshaped).squeeze()
        x_conv_out = x_conv_out.reshape(slen, bsize,x_conv_out.shape[1]).contiguous()
        x_emb = self.highway(x_conv_out)

        return x_emb
        ### END YOUR CODE

