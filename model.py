# La implementación se basa en el tutorial 
# https://medium.com/@sayedebad.777/building-a-transformer-from-scratch-a-step-by-step-guide-a3df0aeb7c9a

import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    '''
    Módulo encargado de obtener el input embedding a partir de los tokens de entrada

    Atributos:
        d_model: tamaño del embedding
        vocab_size: tamaño del vocabulario del tokenizador de entrada
        embedding: capa de embedding
    '''

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    '''
    Módulo encargado de agregar la codificación posicional a los embeddings
    
    Atributos:
        d_model: tamaño del embedding
        seq: largo de la secuencia
        dropout: probabilidad de dropout
        pe: matriz de codificación posicional, con la fórmula correspondiente para cada posición
    '''    

    def __init__(self, d_model: int, seq: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq = seq
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq, d_model)
        pe = torch.zeros(seq, d_model)
        
        # Create a vector of shape (seq)
        position = torch.arange(0, seq, dtype=torch.float).unsqueeze(1) # (seq, 1)
        
        # Create a vector of shape (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_model))
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_model))
        
        # Add a batch dimension to the positional encoding
        pe = pe.unsqueeze(0) # (1, seq, d_model)
        
        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq, d_model)
        return self.dropout(x)

class LayerNormalization(nn.Module):
    '''
    Módulo encargado de aplicar la Normalización por capa
    
    '''
    def __init__(self, features: int, eps:float=10**-6) -> None:
        '''
        features: tamaño del vector de entrada
        eps: valor pequeño para evitar dividir por cero
        alpha: parámetro aprendible para escalar la salida
        bias: parámetro aprendible para escalar la salida
        '''
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features)) 
        self.bias = nn.Parameter(torch.zeros(features)) 

    def forward(self, x):
        # x: (batch, seq, hidden_size)
        # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    

class FeedForwardBlock(nn.Module):
    '''
    Red neuronal feed forward completamente conectada
    
    Atributos:
        d_model: tamaño del embedding (tamaño de la capa de entrada y salida)
        d_ff: tamaño de la capa oculta
        dropout: probabilidad de dropout
        linear_1: capa lineal 1
        dropout: dropout
        linear_2: capa lineal 2
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2

    def forward(self, x):
        # (batch, seq, d_model) --> (batch, seq, d_ff) --> (batch, seq, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    

class MultiHeadAttentionBlock(nn.Module):
    '''
    Bloque encargado de aplicar atención con múltiples cabezales.
    Será usado en el Encoder y en el Decoder
    
    Atributos:
        d_model: tamaño del embedding
        h: número de cabezales
        d_k: tamaño del vector visto por cada cabeza
        w_q: matriz de consultas
        w_k: matriz de clave
        w_v: matriz de valor
        w_o: matriz de salida
        dropout: probabilidad de dropout
    '''

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # (batch, h, seq, d_k) --> (batch, h, seq, seq)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq, seq) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq, seq) --> (batch, h, seq, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (batch, seq, d_model) --> (batch, seq, d_model)
        key = self.w_k(k) # (batch, seq, d_model) --> (batch, seq, d_model)
        value = self.w_v(v) # (batch, seq, d_model) --> (batch, seq, d_model)

        # (batch, seq, d_model) --> (batch, seq, h, d_k) --> (batch, h, seq, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq, d_k) --> (batch, seq, h, d_k) --> (batch, seq, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Multiply by Wo
        # (batch, seq, d_model) --> (batch, seq, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
        '''
        Módulo encargado de aplicar la conexión residual a la salida de un bloque,
        sumando la entrada con la salida del bloque y aplicando dropout y normalización

        Atributos:
            features: tamaño del vector de entrada
            dropout: probabilidad de dropout
            norm: capa de normalización        
        '''
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderBlock(nn.Module):
    '''
    Bloque del Encoder
    
    Atributos:
        features: tamaño del vector de entrada
        self_attention_block: bloque de atención
        feed_forward_block: red neuronal feed forward completamente conectada
        residual_connections: conexiones residuales 
            (una después de la atención y otra después de la red neuronal)
    '''
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    '''
    Encoder, compuesto por varios bloques de Encoder
    
    Atributos:
        features: tamaño del vector de entrada
        layers: bloques del Encoder
        norm: capa normalización para aplicar al final del Encoder
    '''

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    '''
    Bloque del Decoder
    
    Atributos:
        features: tamaño del vector de entrada
        self_attention_block: bloque de atención
        cross_attention_block: bloque de atención cruzada
        feed_forward_block: red neuronal feed forward completamente conectada
        residual_connections: conexiones residuales 
            (una después de la atención simple, otra después de la atención cruzada y otra después de la red neuronal)
    '''
    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module):
    '''
    Decoder, compuesto por varios bloques de Decoder
    
    Atributos:
        features: tamaño del vector de entrada
        layers: bloques del Decoder
        norm: capa normalización para aplicar al final del Decoder
    '''

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    '''
    Capa de proyección para transformar el output del Decoder en 
    una distribución de probabilidad sobre el vocabulario del tokenizador de salida
    
    Atributos:
        d_model: tamaño del vector de entrada
        vocab_size: tamaño del vocabulario del tokenizador de salida
        proj: capa lineal para proyectar el output del
            Decoder en una distribución de probabilidad
    '''
    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq, d_model) --> (batch, seq, vocab_size)
        return self.proj(x)
    
class Transformer(nn.Module):
    '''
    Modelo Transformer
    
    Atributos:
        encoder: Encoder
        decoder: Decoder
        src_embed: capa de embedding para los tokens de entrada
        tgt_embed: capa de embedding para los tokens de salida
        src_pos: capa de codificación posicional para los tokens de entrada
        tgt_pos: capa de codificación posicional para los tokens de salida
        projection_layer: capa de proyección
    '''
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # (batch, seq, d_model)
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)
    
    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # (batch, seq, d_model)
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        # (batch, seq, vocab_size)
        return self.projection_layer(x)

def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq: int, tgt_seq: int, 
                      d_model: int=512, N_encoder: int=6, N_decoder: int=6, h: int=8, 
                      dropout: float=0.1, d_ff: int=2048) -> Transformer:
    # Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq, dropout)
    
    # Create the encoder blocks
    encoder_blocks = []
    for _ in range(N_encoder):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    # Create the decoder blocks
    decoder_blocks = []
    for _ in range(N_decoder):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)
    
    # Create the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
    
    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    # Create the transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return transformer