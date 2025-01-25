import json
import pandas as pd
import openai
import os
import numpy as np
import re
import time
import ssl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor

def configure_ssl_proxy():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    openai.verify_ssl_certs = False

    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

def load_api_key(file_path="Key.txt"):
    try:
        with open(file_path, "r") as f:
            api_key = f.read().strip()

        if not (api_key.startswith('sk-') or api_key.startswith('sk-proj-')):
            print("Warning: API key format may be incorrect")
            print(f"API key prefix: {api_key[:7]}")
            print("Please check the content of the Key.txt file")
            exit(1)

        openai.api_key = api_key

    except Exception as e:
        print(f"Error reading API key: {e}")
        exit(1)

    try:
        response = openai.Model.list()
        print("API key validation successful!")
    except Exception as e:
        print(f"API call error: {e}")
        exit(1)

def get_label(prompt, max_retries=3):
    system_prompt = (
        "You are an expert in performance prediction for nitride materials. "
        "You have to give specific numbers,Please provide the Band Gap and Predicted Formation Energy in the following format:\n"
        "\"Band Gap: <value> eV; Predicted Formation Energy: <value> eV/atom.\"\n"
        "Ensure that negative values are correctly represented."
    )
    for attempt in range(max_retries):
        try:
            response = openai.ChatCompletion.create(
                model="ft:gpt-3.5-turbo-0125:personal:samantha-test:AoMIhBbl",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )
            raw_label = response['choices'][0]['message']['content'].strip()
            band_gap, formation_energy = parse_label(raw_label)
            if band_gap is not None and formation_energy is not None:
                return band_gap, formation_energy, raw_label
            else:
                print(f"Parsing failed, trying to retrieve label again (attempt {attempt + 1}/{max_retries})")
        except openai.error.OpenAIError as e:
            print(f"OpenAI API error for prompt: {prompt}\nError: {e}")
        except Exception as e:
            print(f"Unexpected error for prompt: {prompt}\nError: {e}")
        time.sleep(2)
    print(f"Unable to retrieve valid label, skipping this sample: {prompt}")
    return None, None, None

def parse_label(raw_label):
    band_gap = None
    formation_energy = None
    try:
        band_gap_match = re.search(r'Band\s*Gap\s*[:\-]?\s*([-\d.]+)\s*eV', raw_label, re.IGNORECASE)
        formation_energy_match = re.search(
            r'Predicted\s*Formation\s*Energy\s*[:\-]?\s*([-\d.]+)\s*eV/atom', raw_label, re.IGNORECASE)

        if band_gap_match:
            band_gap = float(band_gap_match.group(1))
        if formation_energy_match:
            formation_energy = float(formation_energy_match.group(1))
    except Exception as e:
        print(f"Error parsing label: {raw_label}\nError: {e}")

    if band_gap is None or formation_energy is None:
        print(f"Unable to parse label: {raw_label}")

    return band_gap, formation_energy

def load_data(filepath):
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                json_obj = json.loads(line)
                messages = json_obj.get('messages', [])
                user_content = ""
                for message in messages:
                    if message.get('role') == 'user':
                        user_content = message.get('content', "")
                data.append({
                    'input': user_content,
                    'band_gap': None,
                    'formation_energy': None,
                    'raw_label': None
                })
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s\.-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def active_learning(df, target_labels=100, initial_samples=10, iterations=20, samples_per_iteration=5, sleep_time=1):
    labeled_data = pd.DataFrame(columns=['messages'])
    unlabeled_data = df.copy()

    system_message = {
        "role": "system",
        "content": "You are an expert in performance prediction for nitride materials."
    }

    if initial_samples > len(unlabeled_data):
        initial_samples = len(unlabeled_data)
    initial_labeled = unlabeled_data.sample(n=initial_samples, random_state=42).copy()
    print(f"Generating labels for initial {initial_samples} samples...")

    for idx, row in initial_labeled.iterrows():
        prompt = row['input']
        band_gap, formation_energy, raw_label = get_label(prompt)
        if band_gap is not None and formation_energy is not None:
            messages = [
                system_message,
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": raw_label}
            ]
            labeled_data = pd.concat([labeled_data, pd.DataFrame([{"messages": messages}])], ignore_index=True)
            print(f"Initial sample {idx} successfully labeled: Band Gap={band_gap}, Formation Energy={formation_energy}")
            unlabeled_data = unlabeled_data.drop(idx)
        else:
            print(f"Initial sample {idx} failed to get label.")
        time.sleep(sleep_time)

    unlabeled_data = unlabeled_data.reset_index(drop=True)
    print(f"Initial labeled samples: {len(labeled_data)}")
    print(f"Initial unlabeled samples: {len(unlabeled_data)}")

    if len(labeled_data) >= target_labels:
        print(f"Target label count reached: {target_labels}")
        return labeled_data

    labeled_inputs = labeled_data['messages'].apply(lambda msgs: msgs[1]['content'])
    unlabeled_data['input'] = unlabeled_data['input'].apply(preprocess_text)

    vectorizer = TfidfVectorizer(stop_words=None, ngram_range=(1, 2), max_features=1000)
    X_train = vectorizer.fit_transform(labeled_inputs)
    try:
        y_train_band_gap = labeled_data['messages'].apply(lambda msgs: float(re.search(r'Band\s*Gap\s*:\s*([-\d.]+)\s*eV', msgs[2]['content'], re.IGNORECASE).group(1)))
        y_train_formation_energy = labeled_data['messages'].apply(lambda msgs: float(re.search(r'Predicted\s*Formation\s*Energy\s*:\s*([-\d.]+)\s*eV/atom', msgs[2]['content'], re.IGNORECASE).group(1)))
    except AttributeError as e:
        print(f"Regex matching failed, label format may be incorrect.\nError: {e}")
        exit(1)

    model_band_gap = RandomForestRegressor(n_estimators=100, random_state=42)
    model_band_gap.fit(X_train, y_train_band_gap)

    model_formation_energy = RandomForestRegressor(n_estimators=100, random_state=42)
    model_formation_energy.fit(X_train, y_train_formation_energy)

    for iteration in range(iterations):
        if len(labeled_data) >= target_labels:
            print(f"Target label count reached: {target_labels}")
            break
        if len(unlabeled_data) == 0:
            print("All data have been labeled.")
            break

        X_unlabeled = vectorizer.transform(unlabeled_data['input'])

        preds_band_gap = model_band_gap.predict(X_unlabeled)
        preds_formation_energy = model_formation_energy.predict(X_unlabeled)

        preds_band_gap_all = np.array([tree.predict(X_unlabeled) for tree in model_band_gap.estimators_])
        variance_band_gap = preds_band_gap_all.var(axis=0)

        preds_formation_energy_all = np.array([tree.predict(X_unlabeled) for tree in model_formation_energy.estimators_])
        variance_formation_energy = preds_formation_energy_all.var(axis=0)

        uncertainty = variance_band_gap + variance_formation_energy

        query_indices = uncertainty.argsort()[-samples_per_iteration:]
        query_samples = unlabeled_data.iloc[query_indices].copy()

        print(f"Iteration {iteration + 1}: Selecting {len(query_samples)} most uncertain samples for labeling...")

        new_labeled = []
        success_indices = []
        for idx, row in query_samples.iterrows():
            prompt = row['input']
            band_gap, formation_energy, raw_label = get_label(prompt)
            if band_gap is not None and formation_energy is not None:
                messages = [
                    system_message,
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": raw_label}
                ]
                new_labeled.append({"messages": messages})
                success_indices.append(idx)
                print(f"Sample {idx} successfully labeled: Band Gap={band_gap}, Formation Energy={formation_energy}")
            else:
                print(f"Sample {idx} failed to get label.")
            time.sleep(sleep_time)

        if new_labeled:
            labeled_data = pd.concat([labeled_data, pd.DataFrame(new_labeled)], ignore_index=True)
        else:
            print("No new labels were successfully obtained this iteration.")
            break

        if success_indices:
            unlabeled_data = unlabeled_data.drop(success_indices).reset_index(drop=True)
            print(f"Dropped {len(success_indices)} labeled samples.")
        else:
            print("No successfully labeled samples, no samples dropped.")

        print(f"Current labeled samples: {len(labeled_data)} / {target_labels}")

        if len(labeled_data) >= target_labels:
            print(f"Target label count reached: {target_labels}")
            break

        labeled_inputs = labeled_data['messages'].apply(lambda msgs: msgs[1]['content'])

        X_train = vectorizer.transform(labeled_inputs)
        try:
            y_train_band_gap = labeled_data['messages'].apply(lambda msgs: float(re.search(r'Band\s*Gap\s*:\s*([-\d.]+)\s*eV', msgs[2]['content'], re.IGNORECASE).group(1)))
            y_train_formation_energy = labeled_data['messages'].apply(lambda msgs: float(re.search(r'Predicted\s*Formation\s*Energy\s*:\s*([-\d.]+)\s*eV/atom', msgs[2]['content'], re.IGNORECASE).group(1)))
        except AttributeError as e:
            print(f"Regex matching failed, label format may be incorrect.\nError: {e}")
            exit(1)

        model_band_gap.fit(X_train, y_train_band_gap)
        model_formation_energy.fit(X_train, y_train_formation_energy)

    return labeled_data

if __name__ == "__main__":
    configure_ssl_proxy()
    load_api_key("Key.txt")

    filepath = 'high_loss_validation_data.jsonl'
    df = load_data(filepath)

    print(f"Total data: {len(df)} samples")

    labeled_data = active_learning(
        df,
        target_labels=100,
        initial_samples=10,
        iterations=20,
        samples_per_iteration=5,
        sleep_time=1
    )

    try:
        with open('expanded_dataset.jsonl', 'w', encoding='utf-8') as f:
            for _, row in labeled_data.iterrows():
                json_line = json.dumps(row['messages'], ensure_ascii=False)
                f.write(json_line + '\n')
        print("Expanded dataset saved to 'expanded_dataset.jsonl'")
    except Exception as e:
        print(f"Error saving dataset: {e}")
