import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

print("cargando modelo ....")

modelo = ChatterboxMultilingualTTS.from_pretrained(device="cpu")
texto = "Hola, bienvenidos a inteligencia artificial generativa, vamos a generar audios a partir de texto y si el computador aguanta, tambien videos"

audio = modelo.generate(
    texto, 
    language_id="es", 
    exaggeration=0.7, 
    cfg_weight=0.4, 
    temperature=0.8, 
    repetition_penalty=1.2, 
    min_p=0.05, 
    top_p=1.0
)

ta.save("audio.wav", audio, modelo.sr)

print("audio guardaro")