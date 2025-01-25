import json
import openai
import os
import ssl
import re

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

standard_models = {
    'GPT-3.5-Turbo': 'o1-preview',
}

system_message = (
    "You are an expert in performance prediction for nitride materials. "
    "Based on the given material structure description, you must classify the Band Gap and Predicted Formation Energy into the following categories:\n"
    "- Band Gap: 'large' (4-6 eV), 'medium' (2-4 eV), 'small' (0-2 eV)\n"
    "- Predicted Formation Energy: 'large' (3-6 eV/atom), 'medium' (0-3 eV/atom), 'small' (-3-0 eV/atom)\n"
    "Output the categories in the following format:\n"
    "Band Gap Category: <large/medium/small>; Predicted Formation Energy Category: <large/medium/small>."
)

validation_file_path = ''

def extract_categories(response_text, line_number, response_type):
    band_gap_category = None
    predicted_formation_energy_category = None

    band_gap_pattern = r'Band\s*Gap\s*Category\s*[:\-]\s*(large|medium|small)'
    predicted_energy_pattern = r'Predicted\s*Formation\s*Energy\s*Category\s*[:\-]\s*(large|medium|small)'

    band_gap_match = re.search(band_gap_pattern, response_text, re.IGNORECASE)
    if band_gap_match:
        band_gap_category = band_gap_match.group(1).lower()

    predicted_energy_match = re.search(predicted_energy_pattern, response_text, re.IGNORECASE)
    if predicted_energy_match:
        predicted_formation_energy_category = predicted_energy_match.group(1).lower()

    if band_gap_category is None or predicted_formation_energy_category is None:
        print(f"Warning: Line {line_number} of {response_type} response failed to extract all categories.")
        print(f"{response_type.capitalize()} response content: {response_text}")

    return band_gap_category, predicted_formation_energy_category

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

                    existing_band_gap_category = None
                    existing_predicted_energy_category = None
                    if existing_assistant:
                        existing_band_gap_category, existing_predicted_energy_category = extract_categories(
                            existing_assistant, line_number, 'existing'
                        )

                    for model_name, model_id in standard_models.items():
                        new_assistant_response = generate_assistant_response(model_id, messages_to_send)

                        if new_assistant_response is None:
                            print(f"Line {line_number}: {model_name} failed to generate response.")
                            continue

                        new_band_gap_category, new_predicted_energy_category = extract_categories(
                            new_assistant_response, line_number, 'new'
                        )

                        print(f"\n--- Line {line_number}, Model: {model_name} ---")
                        print("Input messages:")
                        for msg in messages_to_send:
                            print(f"{msg.get('role').capitalize()}: {msg.get('content')}")
                        if existing_assistant:
                            print("\nExisting Assistant response:")
                            print(existing_assistant)
                            print("Extracted existing categories:")
                            print(f"  Band Gap Category: {existing_band_gap_category}")
                            print(f"  Predicted Formation Energy Category: {existing_predicted_energy_category}")
                        print("\nNew Assistant response:")
                        print(new_assistant_response)
                        print("Extracted new categories:")
                        print(f"  Band Gap Category: {new_band_gap_category}")
                        print(f"  Predicted Formation Energy Category: {new_predicted_energy_category}")
                        print("--- End ---\n")

                except json.JSONDecodeError as jde:
                    print(f"Error parsing JSON on line {line_number}: {jde}")
                except openai.error.OpenAIError as oe:
                    print(f"Error calling OpenAI API on line {line_number}: {oe}")
                except Exception as ex:
                    print(f"Unknown error on line {line_number}: {ex}")

        print("Validation set processing completed. All results are displayed in the console.")

    except FileNotFoundError:
        print(f"File not found: {validation_file_path}")
    except IOError as io_err:
        print(f"File I/O error: {io_err}")
    except Exception as e:
        print(f"Unknown error during validation set processing: {e}")

if __name__ == "__main__":
    read_and_validate_api_key("key.txt")

    if not os.path.isfile(validation_file_path):
        print(f"Input file does not exist: {validation_file_path}")
        exit(1)

    process_validation_set()
