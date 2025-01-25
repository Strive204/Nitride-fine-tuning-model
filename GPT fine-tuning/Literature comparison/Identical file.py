import json
import openai
import os
import ssl
import csv
import re
from collections import defaultdict

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

openai.verify_ssl_certs = False

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

def read_and_validate_api_key(key_file_path="key.txt"):
    try:
        with open(key_file_path, "r") as f:
            api_key = f.read().strip()

        if not (api_key.startswith('sk-') or api_key.startswith('sk-proj-')):
            print("Warning: API key format may be incorrect")
            print(f"API key prefix: {api_key[:7]}")
            print("Please check the content of the key.txt file")
            exit(1)

        openai.api_key = api_key

    except FileNotFoundError:
        print(f"File not found: {key_file_path}")
        exit(1)
    except Exception as e:
        print(f"Error reading API key: {e}")
        exit(1)

    try:
        response = openai.Model.list()
        print("API key validation successful!")
    except openai.error.AuthenticationError:
        print("API call error: Authentication failed. Please check your API key.")
        exit(1)
    except Exception as e:
        print(f"API call error: {e}")
        exit(1)

fine_tuned_model_id = "ft:gpt-3.5-turbo-0125:personal:samantha-test:AoQoR8Hl"

additional_models = {}

system_message = "You are an expert in performance prediction for nitride materials."

validation_file_path = 'tasks_test.jsonl'

all_models = {'FineTuned': fine_tuned_model_id}
all_models.update(additional_models)

def get_output_file_paths(model_name):
    jsonl_path = f'validation_results_{model_name}.jsonl'
    csv_path = f'results_{model_name}.csv'
    return jsonl_path, csv_path

def extract_values(response_text, line_number, response_type):
    band_gap = None
    predicted_formation_energy = None

    band_gap_patterns = [
        r'Band\s*Gap\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV',
        r'Band\s*Gap\s*[:\-]\s*([-+]?\d*\.?\d+)',
        r'band gap\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV',
        r'bandgap\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV'
    ]
    predicted_energy_patterns = [
        r'Predicted\s*Formation\s*Energy\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV/atom',
        r'Predicted\s*Formation\s*Energy\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV per atom',
        r'predicted formation energy\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV/atom',
        r'predicted formation energy\s*[:\-]\s*([-+]?\d*\.?\d+)\s*eV per atom',
    ]

    for pattern in band_gap_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            band_gap = match.group(1)
            break

    for pattern in predicted_energy_patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            predicted_formation_energy = match.group(1)
            break

    if band_gap is None or predicted_formation_energy is None:
        print(f"Warning: Line {line_number} of {response_type} response failed to extract all values.")
        print(f"{response_type.capitalize()} response content: {response_text}")

    return band_gap, predicted_formation_energy

def generate_assistant_response(model_id, messages):
    try:
        response = openai.ChatCompletion.create(
            model=model_id,
            messages=messages,
            temperature=1.0,
            max_tokens=500
        )
        assistant_message = response["choices"][0]["message"]["content"].strip()
        return assistant_message
    except openai.error.OpenAIError as oe:
        print(f"Error calling OpenAI API: {oe}")
        return None
    except Exception as ex:
        print(f"Unknown error: {ex}")
        return None

def process_validation_set():
    try:
        with open(validation_file_path, 'r', encoding='utf-8') as vf:
            output_files = {}
            csv_writers = {}
            for model_name, model_id in all_models.items():
                jsonl_path, csv_path = get_output_file_paths(model_name)
                of = open(jsonl_path, 'w', encoding='utf-8')
                cf = open(csv_path, 'w', encoding='utf-8', newline='')
                output_files[model_name] = of
                csv_writers[model_name] = csv.writer(cf)

                csv_header = [
                    'Line_Number',
                    'Existing_Band_Gap_eV',
                    'Existing_Predicted_Formation_Energy_eV_per_atom',
                    'New_Band_Gap_eV',
                    'New_Predicted_Formation_Energy_eV_per_atom'
                ]
                csv_writers[model_name].writerow(csv_header)

            for line_number, line in enumerate(vf, start=1):
                try:
                    data = json.loads(line)
                    messages = data.get('messages', [])

                    if not isinstance(messages, list) or not messages:
                        print(f"Line {line_number} missing 'messages' field or content is empty.")
                        continue

                    existing_assistant = None
                    for msg in messages:
                        if msg.get("role") == "assistant":
                            existing_assistant = msg.get("content")
                            break

                    messages_to_send = [msg for msg in messages if msg.get("role") != "assistant"]

                    existing_band_gap = None
                    existing_predicted_energy = None
                    if existing_assistant:
                        existing_band_gap, existing_predicted_energy = extract_values(
                            existing_assistant, line_number, 'existing'
                        )

                    for model_name, model_id in all_models.items():
                        new_assistant_response = generate_assistant_response(model_id, messages_to_send)

                        if new_assistant_response is None:
                            print(f"Line {line_number}: {model_name} failed to generate response.")
                            continue

                        new_band_gap, new_predicted_energy = extract_values(
                            new_assistant_response, line_number, 'new'
                        )

                        output_data = {
                            "messages": messages.copy(),
                            "new_assistant_response": new_assistant_response
                        }

                        if existing_assistant:
                            output_data["existing_assistant_response"] = existing_assistant

                        output_files[model_name].write(json.dumps(output_data, ensure_ascii=False) + '\n')

                        csv_row = [
                            line_number,
                            existing_band_gap if existing_band_gap else '',
                            existing_predicted_energy if existing_predicted_energy else '',
                            new_band_gap if new_band_gap else '',
                            new_predicted_energy if new_predicted_energy else ''
                        ]
                        csv_writers[model_name].writerow(csv_row)

                        print(f"Processed line {line_number} for model: {model_name}.")

                except json.JSONDecodeError as jde:
                    print(f"Error parsing JSON on line {line_number}: {jde}")
                except openai.error.OpenAIError as oe:
                    print(f"Error calling OpenAI API on line {line_number}: {oe}")
                except Exception as ex:
                    print(f"Unknown error on line {line_number}: {ex}")

            for of in output_files.values():
                of.close()
            for cf in csv_writers.values():
                cf.close()

        print(f"Validation set processing completed. Results saved to model-specific JSONL and CSV files.")

    except FileNotFoundError:
        print(f"File not found: {validation_file_path}")
    except IOError as io_err:
        print(f"File I/O error: {io_err}")
    except Exception as e:
        print(f"Unknown error during validation set processing: {e}")

if __name__ == "__main__":
    read_and_validate_api_key("key.txt")

    if not fine_tuned_model_id:
        print("Fine-tuned model ID not set. Please specify 'fine_tuned_model_id' in the code.")
        exit(1)

    if not os.path.isfile(validation_file_path):
        print(f"Input file does not exist: {validation_file_path}")
        exit(1)

    process_validation_set()
