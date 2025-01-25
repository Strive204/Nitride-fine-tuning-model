import os


def merge_jsonl_files(file_list, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in file_list:
            if not os.path.isfile(fname):
                print(f"文件 {fname} 不存在，已跳过。")
                continue
            with open(fname, 'r', encoding='utf-8') as infile:
                for line in infile:

                    line = line.strip()
                    if line:
                        outfile.write(line + '\n')
    print(f"合并完成，输出文件为 {output_file}")


if __name__ == "__main__":

    jsonl_files = [
        'AlN材料.jsonl',
        'BN材料.jsonl',
        'CN材料.jsonl',
        'CoN材料.jsonl',
        'CrN材料.jsonl',
        'FeN材料.jsonl',
        'GaN材料.jsonl',
        'HfN材料.jsonl',
        'InN材料.jsonl',
        'MnN材料.jsonl',
        'MoN材料.jsonl',
        'ScN材料.jsonl',
        'SiN材料.jsonl',
        'TcN材料.jsonl',
        'TiN材料.jsonl',
        'VN材料.jsonl',
        'WN材料.jsonl',
        'ZrN材料.jsonl',
        'NbN材料.jsonl',
        'NiN材料.jsonl',

    ]


    output_filename = ''

    merge_jsonl_files(jsonl_files, output_filename)
