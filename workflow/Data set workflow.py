from mp_api.client import MPRester
import pandas as pd

API_KEY = "vsFs2xazxPxS0IfQPO3oMxgusQQLcd7E"

chemsys = "N-Tc"
fields = [
    'material_id', 'band_gap', 'formation_energy_per_atom',
]
MAX_RESULTS = 150

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        chemsys=chemsys,
        fields=fields
    )

    docs = list(docs)[:MAX_RESULTS]

if not docs:
    print("No materials found.")
else:
    df = pd.DataFrame(docs)

    required_columns = [14, 24, 31]
    max_required = max(required_columns) + 1
    if df.shape[1] < max_required:
        print(f"DataFrame does not have {max_required} columns. Current columns: {df.shape[1]}")
    else:
        df_selected = df.iloc[:, required_columns]

        df_selected.columns = [f'Column_{i+1}' for i in required_columns]

        output_filename = "selected_columns_N_K_limited_to_100.csv"
        df_selected.to_csv(output_filename, index=False)

        print(f"Selected columns successfully saved to {output_filename}")

import pandas as pd
import ast
import re

input_filename = "selected_columns_N_K_limited_to_100.csv"
df = pd.read_csv(input_filename, header=None, names=['material_id', 'formation_energy_per_atom', 'band_gap'])

print("Raw data sample:")
print(df.head())

def extract_mpid(cell):
    if isinstance(cell, str):
        match = re.search(r"mp-\d+", cell, re.IGNORECASE)
        if match:
            return match.group(0).lower()
    return None

df['material_id'] = df['material_id'].apply(extract_mpid)

def parse_numeric_tuple(cell):
    try:
        tuple_val = ast.literal_eval(cell)
        return tuple_val[1]
    except (ValueError, SyntaxError):
        return None

df['formation_energy_per_atom'] = df['formation_energy_per_atom'].apply(parse_numeric_tuple)
df['band_gap'] = df['band_gap'].apply(parse_numeric_tuple)

print("\nProcessed data sample:")
print(df.head())

df.rename(columns={
    'material_id': 'Material ID',
    'band_gap': 'Band Gap (eV)',
    'formation_energy_per_atom': 'Formation Energy per Atom (eV/atom)'
}, inplace=True)

df.dropna(inplace=True)

output_filename = "materials_N_K_limited_to_100_cleaned.csv"
df.to_csv(output_filename, index=False)

print(f"\nData successfully saved to {output_filename}")

import warnings
import csv
import json
from robocrys import StructureCondenser, StructureDescriber
from mp_api.client import MPRester

warnings.filterwarnings("ignore", category=UserWarning, module="pymatgen")

API_KEY = "vsFs2xazxPxS0IfQPO3oMxgusQQLcd7E"

input_csv_file = "materials_N_K_limited_to_100_cleaned.csv"
output_jsonl_file = "Partial material/TcN materials.jsonl"

condenser = StructureCondenser()
describer = StructureDescriber()

try:
    with open(input_csv_file, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        material_rows = [row for row in reader]
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit(1)

try:
    with MPRester(api_key=API_KEY) as mpr, open(output_jsonl_file, "w", encoding="utf-8") as jsonl_file:
        for idx, row in enumerate(material_rows, start=1):
            material_id = row['Material ID'].strip()
            formation_energy = row['Formation Energy per Atom (eV/atom)'].strip()
            band_gap = row['Band Gap (eV)'].strip()

            print(f"Processing {idx}/{len(material_rows)}: {material_id}")

            description_content = ""

            try:
                structure = mpr.get_structure_by_material_id(material_id)
            except Exception as e:
                print(f"Error fetching structure for {material_id}: {e}")
                description_content = f"Error: {e}"
            else:
                try:
                    condensed_structure = condenser.condense_structure(structure)
                except Exception as e:
                    print(f"Error condensing structure for {material_id}: {e}")
                    description_content = f"Error: {e}"
                else:
                    try:
                        description_content = describer.describe(condensed_structure)
                    except Exception as e:
                        print(f"Error generating description for {material_id}: {e}")
                        description_content = f"Error: {e}"

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in performance prediction for nitride materials."
                },
                {
                    "role": "user",
                    "content": f"{description_content} What is the Band Gap and Predicted Formation Energy of this nitride material?"
                },
                {
                    "role": "assistant",
                    "content": f"Band Gap : {band_gap} eV; Predicted Formation Energy : {formation_energy} eV/atom."
                }
            ]

            material_json = {
                "messages": messages
            }

            json_str = json.dumps(material_json, ensure_ascii=False)

            jsonl_file.write(json_str + "\n")

except Exception as e:
    print(f"Error processing materials: {e}")
    exit(1)

print(f"All descriptive JSONL data successfully saved to {output_jsonl_file}")
