from pydub import AudioSegment
from pydub.silence import split_on_silence
import soundfile as sf

def remove_silence(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    chunks = split_on_silence(audio, min_silence_len = 1000, silence_thresh = -48)
    no_silence_audio = AudioSegment.empty()
    index = 0
    output_paths = []
    for chunk in chunks:
        index += 1
        no_silence_audio = chunk
        if no_silence_audio.duration_seconds > 120:
            print(f'export to {output_path + str(index) + '.wav'}')
            no_silence_audio.export(output_path + str(index) + '.wav', format = 'wav')
            output_paths.append(output_path + str(index) + '.wav')
    return output_paths

if __name__ == '__main__':
    print()