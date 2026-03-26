import os
import random
from collections import defaultdict

def generate_librispeech_trials(data_dir, output_path, num_pairs=500):

    speaker_dict = defaultdict(list)
    
    for spk in os.listdir(data_dir):
        spk_path = os.path.join(data_dir, spk)
        if not os.path.isdir(spk_path): continue
        
        for chap in os.listdir(spk_path):
            chap_path = os.path.join(spk_path, chap)
            if not os.path.isdir(chap_path): continue
            
            for file in os.listdir(chap_path):
                if file.endswith(".flac"):
                    # Store the ID: e.g., '84-121123-0000'
                    utt_id = file.replace(".flac", "")
                    speaker_dict[spk].append(utt_id)

    speakers = list(speaker_dict.keys())
    trials = []

    # 2. Generate Positive Pairs (Label 1)
    print(f"Generating {num_pairs} positive trials...")
    for _ in range(num_pairs):
        spk = random.choice(speakers)
        # Ensure speaker has at least 2 utterances
        while len(speaker_dict[spk]) < 2:
            spk = random.choice(speakers)
        utt1, utt2 = random.sample(speaker_dict[spk], 2)
        trials.append(f"1 {utt1} {utt2}")

    # 3. Generate Negative Pairs (Label 0)
    print(f"Generating {num_pairs} negative trials...")
    for _ in range(num_pairs):
        spk1, spk2 = random.sample(speakers, 2)
        utt1 = random.choice(speaker_dict[spk1])
        utt2 = random.choice(speaker_dict[spk2])
        trials.append(f"0 {utt1} {utt2}")

    # 4. Save to file
    random.shuffle(trials)
    with open(output_path, 'w') as f:
        for t in trials:
            f.write(t + "\n")
    print(f"Successfully saved trials to {output_path}")

if __name__ == "__main__":
    # Update this path to your actual dev-clean directory
    DATA_PATH = "/home/soham/garments/preet/here/speech_assi1/speech_preet/data/LibriSpeech/dev-clean"
    generate_librispeech_trials(DATA_PATH, "test.txt")