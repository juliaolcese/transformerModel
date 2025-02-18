from pathlib import Path

def get_config():
    '''
    Configuración del modelo de traducción
    
    Hipermarametros:
    - batch_size: Tamaño del lote
    - num_epochs: Número de épocas
    - lr: Tasa de aprendizaje
    - seq: Longitud máxima de secuencia
    - datasource: Dominio (simplemente para nombrar los archivos)
    - lang_src: Lenguaje de origen
    - lang_tgt: Lenguaje de destino
    - model_folder: Carpeta para guardar los pesos del modelo
    - model_basename: Nombre base del archivo de pesos
    - preload: Cargar pesos preentrenados
    - tokenizer_file: Archivo de tokenización
    - experiment_name: Nombre de la carpeta de experimentos
    - N_encoder: Cantidad de bloques de Encoder
    - N_decoder: Cantidad de bloques de Decoder
    - h: Cantidad de cabezleas de atención        
    - d_model: Dimensión de los embeddings
    '''
    return {
        "batch_size": 10,
        "num_epochs": 50,
        "lr": 0.0001,
        "seq": 3750,
        "datasource": 'satellite',
        "lang_src": "layer_facts", 
        "lang_tgt": "plan",
        "model_folder": "weights",
        "model_basename": "model",
        "preload": "latest",
        "tokenizer_file": "tokenizers/tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
        "N_encoder": 2,
        "N_decoder": 6,
        "h": 2,
        "d_model": 128,
    }

def get_weights_file_path(config, epoch: str):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)

# Find the latest weights file in the weights folder
def latest_weights_file_path(config):
    model_folder = f"{config['datasource']}_{config['model_folder']}"
    model_filename = f"{config['model_basename']}*"
    weights_files = list(Path(model_folder).glob(model_filename))
    if len(weights_files) == 0:
        return None
    weights_files.sort()
    return str(weights_files[-1])