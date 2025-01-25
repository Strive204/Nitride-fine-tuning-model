import openai
import time
import os
import ssl
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
import re
import matplotlib.pyplot as plt
import json

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

openai.verify_ssl_certs = False

os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'

system_message = "You are an expert in performance prediction for nitride materials."

try:
    with open("Key.txt", "r") as f:
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

training_file_name = '最后一次更新/1.jsonl'
validation_file_name = '最后一次更新/2.jsonl'

try:
    with open(training_file_name, 'r', encoding='utf-8') as f:
        training_data = [json.loads(line) for line in f]
    n_training_examples = len(training_data)
    print(f"Training data count: {n_training_examples}")
except Exception as e:
    print(f"Error reading training data: {e}")
    exit(1)

try:
    with open(validation_file_name, 'r', encoding='utf-8') as f:
        validation_data = [json.loads(line) for line in f]
    n_validation_examples = len(validation_data)
    print(f"Validation data count: {n_validation_examples}")
except Exception as e:
    print(f"Error reading validation data: {e}")
    exit(1)

try:
    training_response = openai.File.create(
        file=open(training_file_name, "rb"), purpose="fine-tune"
    )
    training_file_id = training_response["id"]

    validation_response = openai.File.create(
        file=open(validation_file_name, "rb"), purpose="fine-tune"
    )
    validation_file_id = validation_response["id"]

    print("Training file id:", training_file_id)
    print("Validation file id:", validation_file_id)

except Exception as e:
    print(f"Error uploading files: {e}")
    exit(1)

suffix_name = "samantha-test"
n_epochs = 5

try:
    response = openai.FineTuningJob.create(
        training_file=training_file_id,
        validation_file=validation_file_id,
        model="ft:gpt-3.5-turbo-0125:personal:samantha-test:AoMIhBbl",
        suffix=suffix_name,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {"n_epochs": n_epochs},
            },
        },
    )

    job_id = response["id"]
    print("Fine-tuning job created:", job_id)

except Exception as e:
    print(f"Error creating fine-tuning job: {e}")
    exit(1)

max_wait_time = 20000
start_time = time.time()

fine_tuned_model_id = None

while True:
    try:
        response = openai.FineTuningJob.retrieve(job_id)
        status = response["status"]
        print(f"Current status: {status}")

        if status == "succeeded":
            fine_tuned_model_id = response["fine_tuned_model"]
            print(f"Training completed! Model ID: {fine_tuned_model_id}")
            break
        elif status == "failed":
            print("Training failed!")
            print(response.get("error", {}))
            exit(1)
        elif time.time() - start_time > max_wait_time:
            print("Wait time exceeded 1 hour, exiting the program")
            exit(1)

    except Exception as e:
        print(f"Error checking fine-tuning job status: {e}")
        exit(1)

    print("Rechecking after 30 seconds...")
    time.sleep(30)

try:
    response = openai.FineTuningJob.list_events(id=job_id, limit=15000)
    events = response["data"]
    events.reverse()

    print("\nFine-tuning job event log:")
    for event in events:
        print(event["message"])

    loss_pattern = re.compile(
        r"Step (\d+)/\d+: training loss=([\d\.]+)"
        r"(?:, validation loss=([\d\.]+))?"
        r"(?:, full validation loss=([\d\.]+))?"
    )
    steps = []
    training_losses = []
    validation_losses = []
    full_validation_losses = []

    for event in events:
        message = event.get("message", "")
        match = loss_pattern.search(message)
        if match:
            step = int(match.group(1))
            training_loss = float(match.group(2))
            validation_loss = float(match.group(3)) if match.group(3) else None
            full_validation_loss = float(match.group(4)) if match.group(4) else None

            steps.append(step)
            training_losses.append(training_loss)
            validation_losses.append(validation_loss)
            full_validation_losses.append(full_validation_loss)
        else:
            pass

    if steps and training_losses:
        plt.figure(figsize=(12, 8))
        plt.plot(steps, training_losses, marker='o', linestyle='-', color='b', label='Training Loss')

        if any(loss is not None for loss in validation_losses):
            plt.plot(
                [s for s, v in zip(steps, validation_losses) if v is not None],
                [v for v in validation_losses if v is not None],
                marker='x',
                linestyle='--',
                color='r',
                label='Validation Loss'
            )

        if any(loss is not None for loss in full_validation_losses):
            plt.plot(
                [s for s, fv in zip(steps, full_validation_losses) if fv is not None],
                [fv for fv in full_validation_losses if fv is not None],
                marker='s',
                linestyle=':',
                color='g',
                label='Full Validation Loss'
            )

        plt.title('Loss Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('training_validation_loss.png')
        plt.show()

        training_high_loss_threshold = 0.7
        high_loss_steps = [step for step, loss in zip(steps, training_losses) if loss > training_high_loss_threshold]
        print(f"\nTraining steps with loss higher than {training_high_loss_threshold}: {len(high_loss_steps)}")

        high_loss_training_indices = set()
        for step in high_loss_steps:
            data_index = (step - 1) % n_training_examples
            high_loss_training_indices.add(data_index)

        print(f"Number of high-loss training data to extract: {len(high_loss_training_indices)}")

        high_loss_training_data = [training_data[idx] for idx in sorted(high_loss_training_indices)]

        high_loss_training_file_name = 'high_loss_training_data.jsonl'
        try:
            with open(high_loss_training_file_name, 'w', encoding='utf-8') as f:
                for entry in high_loss_training_data:
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"High-loss training data saved to {high_loss_training_file_name}")
        except Exception as e:
            print(f"Error saving high-loss training data: {e}")

        validation_high_loss_threshold = 0.7
        high_loss_validation_steps = []

        for v_loss in validation_losses:
            if v_loss is not None and v_loss > validation_high_loss_threshold:
                high_loss_validation_steps.append(1)

        for fv_loss in full_validation_losses:
            if fv_loss is not None and fv_loss > validation_high_loss_threshold:
                high_loss_validation_steps.append(1)

        total_val_loss_events = len(validation_losses) + len(full_validation_losses)
        if total_val_loss_events == 0:
            print("No validation loss events found.")
            high_loss_validation_indices = set()
        else:
            num_val_rounds = total_val_loss_events / n_validation_examples
            if not num_val_rounds.is_integer():
                print("Warning: Total number of validation loss events cannot be evenly divided by the validation dataset size.")
                num_val_rounds = int(num_val_rounds)
            else:
                num_val_rounds = int(num_val_rounds)
            print(f"Validation rounds: {num_val_rounds}")

            combined_validation_losses = []
            for v_loss in validation_losses:
                if v_loss is not None:
                    combined_validation_losses.append(v_loss)
            for fv_loss in full_validation_losses:
                if fv_loss is not None:
                    combined_validation_losses.append(fv_loss)

            high_loss_validation_indices = set()
            for idx, loss in enumerate(combined_validation_losses):
                if loss > validation_high_loss_threshold:
                    data_index = idx % n_validation_examples
                    high_loss_validation_indices.add(data_index)

            print(f"Number of high-loss validation data to extract: {len(high_loss_validation_indices)}")

        high_loss_validation_data = [validation_data[idx] for idx in sorted(high_loss_validation_indices)]

        high_loss_validation_file_name = '最后一次更新/high_loss_validation_data.jsonl'
        try:
            with open(high_loss_validation_file_name, 'w', encoding='utf-8') as f:
                for entry in high_loss_validation_data:
                    json_line = json.dumps(entry, ensure_ascii=False)
                    f.write(json_line + '\n')
            print(f"High-loss validation data saved to {high_loss_validation_file_name}")
        except Exception as e:
            print(f"Error saving high-loss validation data: {e}")

    else:
        print("No training loss data found. Please check if the event message format is correct.")

except Exception as e:
    print(f"Error getting or processing fine-tuning job events: {e}")
