# Transformer Model
Este repositorio implementa el código para entrenar un modelo Transformer para la tarea de traducción. Se implementan además dos funciones para usar el modelo entrenado para traducir siguiendo una estrategia greedy y beam search.

## Archivos

- `config.py`: Configuración del modelo.
- `dataset.py`: Clases para cargar los datos de entrenamiento y validación.
- `get_translation.py`: Funciones para hacer uso del modelo.
- `model.py`: Definición de la arquitectura del modelo Transformer.
- `train.py`: Código para entrenar el modelo.
- `requirements.txt`: Lista de dependencias necesarias.

## Dependencias

Para instalar las dependencias necesarias, ejecutar:

```bash
pip install -r requirements.txt
```

2. Entrenar el modelo:

```bash
python train.py --train_file <ruta al archivo con datos entrenamiento> --val_file <ruta al archivo con datos de validación>
```
