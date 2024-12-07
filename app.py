import os
import torch
import subprocess
import tempfile
from flask import Flask, request, jsonify, send_file
from transformers import VitsTokenizer, VitsModel, set_seed
import scipy.io.wavfile
from TTS.api import TTS

app = Flask(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LANGUAGE_MAP = {
    'english': {'code': 'eng', 'full_name': 'English'},
    'akan': {'code': 'aka', 'full_name': 'Akan'},
    'ewe': {'code': 'ewe', 'full_name': 'Ewe'},
    'ga': {'code': 'gaa', 'full_name': 'Ga'}
}

def uromanize(input_string, uroman_path):
    """Convert non-Roman strings to Roman using the `uroman` perl package."""
    script_path = os.path.join(uroman_path, "bin", "uroman.pl")
    command = ["perl", script_path]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate(input=input_string.encode())
    if process.returncode != 0:
        raise ValueError(f"Error {process.returncode}: {stderr.decode()}")
    return stdout.decode()[:-1]

def text_to_speech(text, language_code):
    """Generate speech for given text and language"""
    set_seed(555)

    model_name = f"facebook/mms-tts-{language_code}"
    tokenizer = VitsTokenizer.from_pretrained(model_name)
    model = VitsModel.from_pretrained(model_name)

    if tokenizer.is_uroman:
        text = uromanize(text, os.environ["UROMAN"])

    inputs = tokenizer(text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.waveform[0], model.config.sampling_rate

def text_to_speech_with_voice_cloning(text, language_code):
    """
    Generate speech with voice cloning using TTS library
    
    Args:
        text (str): Input text to convert to speech
        target_lang (str): Target language from LANGUAGE_MAP
    
    Returns:
        str: Path to generated wav file
    """
    
    voices_dir = './voices' 
    speaker_wav = os.path.join(voices_dir, f"{language_code}.wav")
    
    if not os.path.exists(speaker_wav):
        raise ValueError(f"Voice file not found: {speaker_wav}")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output:
        output_path = temp_output.name
    
    tts = TTS(
        model_name=f"tts_models/{language_code}/fairseq/vits", 
        progress_bar=False
    ).to(DEVICE)
    
    tts.tts_with_vc_to_file(
        text=text,
        speaker_wav=speaker_wav,
        file_path=output_path
    )
    
    return output_path

@app.route('/tts', methods=['POST'])
def generate_speech():
    data = request.json
    text = data.get('text', '').lower()
    target_lang = data.get('target_lang', '').lower()

    if not text or not target_lang:
        return jsonify({"error": "Text and target language are required"}), 400

    if target_lang not in LANGUAGE_MAP:
        return jsonify({"error": f"Unsupported language. Supported: {list(LANGUAGE_MAP.keys())}"}), 400

    try:
        # with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
        #     language_code = LANGUAGE_MAP[target_lang]['code']
        #     waveform, sample_rate = text_to_speech(text, language_code)
            
        #     scipy.io.wavfile.write(temp_file.name, sample_rate, waveform.numpy())
        #     temp_file_path = temp_file.name
        
        language_code = LANGUAGE_MAP[target_lang]['code']
        output_path = text_to_speech_with_voice_cloning(text, language_code)

        response = send_file(
            # temp_file_path,
            output_path, 
            mimetype='audio/wav', 
            as_attachment=True, 
            download_name=f'yaa_ai_{target_lang}_tts.wav'
        )
        
        def remove_temp_file():
            os.unlink(temp_file_path)
        response.call_on_close(remove_temp_file)

        return response

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))