import json
import os

# Carrega arquivo JSON com os prompts
def load_prompts_json(input_file_path):
    with open(input_file_path) as file:
        return json.load(file)
    
def list_files_from_directory(directory):
    files_names = []
    file_list = os.scandir(directory)
    for file in file_list:
        if file.is_file():
            files_names.append(file.name)
    return files_names