import os
import json
import whisper
import argparse
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="whisper")

# Constants
MODEL_NAME = 'large-v2'
LANGUAGE = 'el'
WORD_TIMESTAMPS = True

def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int((secs - int(secs)) * 1000)
    return f"{hours:02}:{minutes:02}:{int(secs):02},{milliseconds:03}"

def generate_srt_per_word(json_data, output_path):
    with open(output_path, 'w', encoding='utf-8') as srt_file:
        index = 1
        for segment in json_data['segments']:
            for word in segment['words']:
                start = format_timestamp(word['start'])
                end = format_timestamp(word['end'])
                srt_file.write(f"{index}\n{start} --> {end}\n{word['word']}\n\n")
                index += 1

def main():
    parser = argparse.ArgumentParser(description="Transcribe audio using Whisper model")
    parser.add_argument('--input', type=str, required=True, help='Path to the media file to transcribe')
    parser.add_argument('--output', type=str, required=True, help='Directory to save the output files')
    args = parser.parse_args()

    media_file = args.input
    output_dir = args.output

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load Whisper model
    model = whisper.load_model(MODEL_NAME)

    # Transcribe audio file
    result = model.transcribe(
        media_file,
        language=LANGUAGE,
        word_timestamps=WORD_TIMESTAMPS
    )

    # Save JSON result
    json_output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '.json')
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4)

    # Create SRT per sentence
    srt_output_path_sentence = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '_sentence.srt')
    with open(srt_output_path_sentence, 'w', encoding='utf-8') as srt_file:
        for i, segment in enumerate(result['segments']):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text'].strip()
            srt_file.write(f"{i + 1}\n{start_time} --> {end_time}\n{text}\n\n")

    # Create SRT per word
    srt_output_path_word = os.path.join(output_dir, os.path.splitext(os.path.basename(media_file))[0] + '_word.srt')
    generate_srt_per_word(result, srt_output_path_word)

    print(f"Transcription completed. Results saved in the directory '{output_dir}'.")

if __name__ == "__main__":
    main()
