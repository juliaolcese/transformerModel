import torch
from dataset import causal_mask
import torch.nn.functional as F
import math

def greedy_translation(src_text, tokenizer_src, tokenizer_tgt, config, model, device):
    # Implementación siguiendo el tutorial
    # https://medium.com/@sayedebad.777/training-a-transformer-model-from-scratch-25bb270f5888
    '''
    Obtiene la traducción de una secuencia con greedy decode.
    
    Parámetros:
    src_text: Texto de origen
    tokenizer_src: Tokenizador de entrada
    tokenizer_tgt: Tokenizador de salida
    config: Configuración del modelo
    model: Modelo de traducción
    device: Dispositivo
    '''
    with torch.no_grad():
        seq = config["seq"]
        sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        enc_input_tokens = tokenizer_src.encode(src_text).ids
        enc_num_padding_tokens = seq - len(enc_input_tokens) - 2

        if enc_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device) # (1, 1, seq)
        encoder_input = encoder_input.unsqueeze(0).to(device)

        # Precompute the encoder output and reuse it for every step
        encoder_output = model.encode(encoder_input, encoder_mask)
        # Initialize the decoder input with the sos token
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(encoder_input).to(device)


        while True:
            if decoder_input.size(1) == seq:
                break

            # build mask for target
            decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

            # get next token
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(encoder_input).fill_(next_word.item()).to(device)], dim=1
            )

            if next_word == eos_idx:
                break

        model_out = decoder_input.squeeze(0)
        model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

    return model_out_text


def beam_translation(src_text, tokenizer_src, tokenizer_tgt, config, model, device, 
                     beam_size=4, length_norm_coefficient=0.6):
    # Implementación siguiendo el tutorial
    # https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Transformers

    '''
    Obtiene la traducción de una secuencia con beam search.
    
    Parámetros:
    src_text: Texto de origen
    tokenizer_src: Tokenizador de entrada
    tokenizer_tgt: Tokenizador de salida
    config: Configuración del modelo
    model: Modelo de traducción
    device: Dispositivo
    beam_size: Tamaño del beam
    length_norm_coefficient: Coeficiente de normalización de longitud
    '''
    with torch.no_grad():
        k = beam_size
            
        vocab_size = tokenizer_tgt.get_vocab_size()

        seq = config["seq"]
        sos_token = torch.tensor([tokenizer_tgt.token_to_id("[SOS]")], dtype=torch.int64)
        eos_token = torch.tensor([tokenizer_tgt.token_to_id("[EOS]")], dtype=torch.int64)
        pad_token = torch.tensor([tokenizer_tgt.token_to_id("[PAD]")], dtype=torch.int64)

        enc_input_tokens = tokenizer_src.encode(src_text).ids
        enc_num_padding_tokens = seq - len(enc_input_tokens) - 2

        if enc_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        encoder_input = torch.cat(
            [
                sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                eos_token,
                torch.tensor([pad_token] * enc_num_padding_tokens, dtype=torch.int64),
            ],
            dim=0,
        )

        encoder_mask = (encoder_input != pad_token).unsqueeze(0).unsqueeze(0).int().to(device) # (1, 1, seq)
        encoder_input = encoder_input.unsqueeze(0).to(device)

        encoder_output = model.encode(encoder_input, encoder_mask)
    
        sos_idx = tokenizer_tgt.token_to_id('[SOS]')
        eos_idx = tokenizer_tgt.token_to_id('[EOS]')

        translations = torch.tensor([[sos_idx]]).to(device)
        translations_scores = torch.zeros(1).to(device)
        
        full_translations = list()
        full_translations_scores = list()

        step = 1 
        
        while True:
            # build mask for target
            decoder_mask = causal_mask(translations.size(1)).type_as(encoder_mask).to(device)

            # calculate output
            out = model.decode(encoder_output, encoder_mask, translations, decoder_mask)

            # get next token
            scores = model.project(out[:, -1, :])
            scores = F.log_softmax(scores, dim=-1)
            scores = translations_scores.unsqueeze(1) + scores

            top_k_translations_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            top_k_translations = torch.cat([translations[prev_word_indices], next_word_indices.unsqueeze(1)],
                                dim=1)  # (k, step + 1)

            complete = next_word_indices == eos_idx  # (k), bool

            full_translations.extend(top_k_translations[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            full_translations_scores.extend((top_k_translations_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(full_translations) >= k:
                break

            # Else, continue with incomplete hypotheses
            translations = top_k_translations[~complete]  # (s, step + 1)
            translations_scores = top_k_translations_scores[~complete]  # (s)

            # Stop if things have been going on for too long
            if step > seq:
                break
        
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(full_translations) == 0:
            full_translations = translations.tolist()
            full_translations_scores = translations_scores.tolist()

        # Decode the hypotheses
        all_translations = list()
        for i in range(len(full_translations)):
            translation = tokenizer_tgt.decode(full_translations[i])
            all_translations.append({"translation": translation, "score": full_translations_scores[i]})


        # Find the best scoring completed hypothesis
        i = full_translations_scores.index(max(full_translations_scores))
        best_hypothesis = all_translations[i]["translation"]

        return best_hypothesis, all_translations