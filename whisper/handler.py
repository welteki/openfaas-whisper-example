import os
import tempfile
from urllib.request import urlretrieve

import whisper

def handle(event, context):
    models_cache = os.getenv("MODELS_CACHE", "/tmp/models")
    model_size = os.getenv("MODEL_SIZE", "tiny.en")

    url = str(event.body, "UTF-8")
    audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=True)
    urlretrieve(url, audio.name)

    model = whisper.load_model(name=model_size, download_root=models_cache)
    result = model.transcribe(audio.name)
    
    return (result["text"], 200, {'Content-Type': 'text/plain'})

