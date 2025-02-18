# Implementación basada en el tutorial de 
# https://medium.com/@sayedebad.777/training-a-transformer-model-from-scratch-25bb270f5888

from model import build_transformer
from get_translation import greedy_decode
from dataset import TranslationDataset, TranslationDatasetWithOps
from config import get_config, get_weights_file_path, latest_weights_file_path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

import warnings
from tqdm import tqdm
import os
from pathlib import Path
import json
import argparse

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, 
                   device, print_msg, global_step, writer, num_examples=2):
    '''
    Realiza la validación del modelo en el conjunto de validación
    
    Parámetros:
    model: Modelo a validar
    validation_ds: Conjunto de validación
    tokenizer_src: Tokenizador de la entrada
    tokenizer_tgt: Tokenizador de la salida
    max_len: Longitud máxima de la secuencia
    device: Dispositivo a utilizar
    '''
    model.eval()
    count = 0

    source_texts = []
    expected = []
    predicted = []

    try:
        # get the console window width
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        # If we can't get the console width, use 80 as default
        console_width = 80

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device) # (b, seq)
            encoder_mask = batch["encoder_mask"].to(device) # (b, 1, 1, seq)

            # check that the batch size is 1
            assert encoder_input.size(
                0) == 1, "Batch size must be 1 for validation"

            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_tgt, max_len, device)

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)
            
            # Print the source, target and model output
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break
    
    if writer:
        metric = torchmetrics.CharErrorRate()
        cer = metric(predicted, expected)
        writer.add_scalar('validation cer', cer, global_step)
        writer.flush()

        # Compute the word error rate
        metric = torchmetrics.WordErrorRate()
        wer = metric(predicted, expected)
        writer.add_scalar('validation wer', wer, global_step)
        writer.flush()

        # Compute the BLEU metric
        metric = torchmetrics.BLEUScore()
        bleu = metric(predicted, expected)
        writer.add_scalar('validation BLEU', bleu, global_step)

def get_all_sentences(ds, lang):
    '''
    Obtiene todas las oraciones de un idioma en un conjunto de datos
    
    Parámetros:
    ds: Conjunto de datos
    lang: Idioma a extraer
    '''
    for item in ds:
        yield item[lang]

def get_or_build_tokenizer(config, ds, lang):
    '''
    Obtener los tokenizadores de un idioma o construirlos si no existen
    
    Parámetros:
    config: Configuración del modelo
    ds: Conjunto de datos
    lang: Idioma del tokenizador
    '''
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Si el tokenizador no existe, define el vocabulario separando por espacios
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_ds(config, train_file, val_file, use_eval = False, with_operators = False):
    '''
    Obtiene los conjuntos de entrenamiento y validación para el modelo
    
    Parámetros:
    config: Configuración del modelo
    train_file: Archivo con ejemplos de entrenamiento
    val_file: Archivo con ejemplos de validación
    use_eval: Si se debe usar el conjunto de validación, o si se debe dividir el conjunto de entrenamiento
    with_operators: Si se deben utilizar los Buenos y Malos Operadores en el conjunto de validación
        (útil cuando se optimiza el modelo)
    '''
    with open(train_file, "r") as file:
        ds_raw = json.load(file)

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer (config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    if use_eval:
        train_ds_raw = ds_raw
        with open(val_file, "r") as file:
            val_ds_raw = json.load(file)
    else:
    # Keep 90% for training, 10% for validation
        train_ds_size = int(0.9 * len(ds_raw))
        val_ds_size = len(ds_raw) - train_ds_size
        train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = TranslationDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, 
                                  config['lang_src'], config['lang_tgt'], config['seq'])
    if with_operators:
        val_ds = TranslationDatasetWithOps(val_ds_raw, tokenizer_src, tokenizer_tgt, 
                                           config['lang_src'], config['lang_tgt'], config['seq'])
    else:
        val_ds = TranslationDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], 
                                    config['lang_tgt'], config['seq'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item[config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item[config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f'Max length of source sentence: {max_len_src}')
    print(f'Max length of target sentence: {max_len_tgt}')
    

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocab_src_len, vocab_tgt_len):
    '''
    Obtiene el modelo de traducción
    
    Parámetros:
    config: Configuración del modelo
    vocab_src_len: Longitud del vocabulario del tokenizador de entrada
    vocab_tgt_len: Longitud del vocabulario del tokenizador de salida    
    '''
    model = build_transformer(vocab_src_len, vocab_tgt_len, config["seq"], config['seq'], d_model=config['d_model'], 
                              N_encoder=config["N_encoder"], N_decoder=config["N_decoder"],h=config['h'])        
    
    print(f'h: {config["h"]}')
    print(f'N_encoder: {config["N_encoder"]}')
    print(f'N_decoder: {config["N_decoder"]}')
    print(f'd_model: {config["d_model"]}')
    return model

def train_model(config, train_file, val_file):
    '''
    Entrena el modelo de traducción
    
    Parámetros:
    config: Configuración del modelo
    '''
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    device = torch.device(device)

    # Make sure the weights folder exists
    Path(f"{config['datasource']}_{config['model_folder']}").mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config, train_file, val_file, use_eval = True, with_operators = True)
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    # Tensorboard
    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # If the user specified a model to preload before training, load it
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = latest_weights_file_path(config) if preload == 'latest' else get_weights_file_path(config, preload) if preload else None
    if model_filename:
        print(f'Preloading model {model_filename}')
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer_state_dict'])
        global_step = state['global_step']
    else:
        print('No model to preload, starting from scratch')

    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer_src.token_to_id('[PAD]'), label_smoothing=0.1).to(device)

    for epoch in range(initial_epoch, config['num_epochs']):
        torch.cuda.empty_cache()
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f"Processing Epoch {epoch:02d}")
        for batch in batch_iterator:

            encoder_input = batch['encoder_input'].to(device) # (b, seq)
            decoder_input = batch['decoder_input'].to(device) # (B, seq)
            encoder_mask = batch['encoder_mask'].to(device) # (B, 1, 1, seq)
            decoder_mask = batch['decoder_mask'].to(device) # (B, 1, seq, seq)

            # Run the tensors through the encoder, decoder and the projection layer
            encoder_output = model.encode(encoder_input, encoder_mask) # (B, seq, d_model)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask) # (B, seq, d_model)
            proj_output = model.project(decoder_output) # (B, seq, vocab_size)

            # Compare the output with the label
            label = batch['label'].to(device) # (B, seq)

            # Compute the loss using a simple cross entropy
            loss = loss_fn(proj_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            # Log the loss
            writer.add_scalar('train loss', loss.item(), global_step)
            writer.flush()

            # Backpropagate the loss
            loss.backward()

            # Update the weights
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq'], device, lambda msg: batch_iterator.write(msg), global_step, writer)

        # Save the model at the end of every epoch
        model_filename = get_weights_file_path(config, f"{epoch:02d}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'global_step': global_step
        }, model_filename)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser(description="Entrea un modelo de traducción")
    parser.add_argument("--train_file", type=str, help="Archivo con ejemplos de entrenamiento")
    parser.add_argument("--val_file", type=str, help="Archivo con ejemplos de validación")
    
    args = parser.parse_args()
    
    config = get_config()
    train_model(config, args.train_file, args.val_file)