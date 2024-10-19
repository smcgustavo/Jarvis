import speech_recognition as sr
import os
from dotenv import load_dotenv
import google.generativeai as genai
from google.cloud import texttospeech
from pydub import AudioSegment
import simpleaudio as sa
import io

def transcrever_audio(recognizer):
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Ready...")
        
        # Escuta o áudio do microfone
        audio = recognizer.listen(source)
        try:
            texto = recognizer.recognize_google(audio, language="pt-BR")
            return texto
        except sr.UnknownValueError:
            return False
        except sr.RequestError as e:
            return False

def checkJarvis(text):
    if "jarvis" in text.lower().split(' '):
        return True
    return False

def reproduzir_audio_binario(dados_binarios):
    # Converte os dados binários em um arquivo de áudio utilizando um buffer
    audio = AudioSegment.from_file(io.BytesIO(dados_binarios), format="mp3")

    # Exporta o áudio para um formato que pode ser lido pelo simpleaudio (WAV)
    audio_wav = io.BytesIO()
    audio.export(audio_wav, format="wav")
    audio_wav.seek(0)  # Move para o início do arquivo

    # Carrega e reproduz o áudio com simpleaudio
    wave_obj = sa.WaveObject.from_wave_file(audio_wav)
    play_obj = wave_obj.play()

    # Aguarda a conclusão da reprodução
    play_obj.wait_done()


def speakJarvis(text, jarvis):
    input = texttospeech.SynthesisInput(text=text)
    
    voice = texttospeech.VoiceSelectionParams(
        language_code="pt-BR",
        name= "pt-BR-Neural2-B",
        ssml_gender = texttospeech.SsmlVoiceGender.MALE
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding = texttospeech.AudioEncoding.MP3
    )
    response = jarvis.synthesize_speech(
        request = {
            "input": input,
            "voice": voice,
            "audio_config": audio_config
        }
    )
    return response

def mainLoop(model, recognizer, prePrompt, jarvis):
    while True:
        text = transcrever_audio(recognizer)
        if text == False:
            continue
        if not checkJarvis(text):
            continue
        speech = prePrompt + text
        response = model.generate_content(speech)
        response = response.to_dict()['candidates'][0]['content']['parts'][0]['text']
        print(response)
        audio = speakJarvis(response, jarvis)
        reproduzir_audio_binario(audio.audio_content)
        

def startJarvis():
    prePrompt = ""
    with open('prompt.txt', "r") as f:
        prePrompt = f.read()
    load_dotenv()
    # Recognizer start
    recognizer = sr.Recognizer()
    
    # Gemini Start
    genai.configure(api_key= os.getenv('KEY'))
    model = genai.GenerativeModel("gemini-1.5-flash")
    jarvis = texttospeech.TextToSpeechClient()

    # Return preprompt
    return model, recognizer, prePrompt, jarvis


model, recognizer, prePrompt, jarvis = startJarvis()

mainLoop(model, recognizer, prePrompt, jarvis)