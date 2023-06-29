import pandas as pd
from label_studio_sdk import Client
import base64
import time
from tqdm import tqdm
import argparse
from dotenv import load_dotenv, dotenv_values
import os

load_dotenv()

PARSER = argparse.ArgumentParser(description="Script to upload images to labelstudio")

PARSER.add_argument(
    "--data-path",
    metavar="d",
    type=str,
    nargs="?",
    help="Path to the csv data file",
)


PARSER.add_argument(
    "--url",
    metavar="u",
    type=str,
    nargs="?",
    help="url of deployed labelstudio instance",
)



ARGS = PARSER.parse_args()


# Define the URL where Label Studio is accessible and the API key for your user account
LABEL_STUDIO_URL = ARGS.url
API_KEY = os.getenv("LABEL_STUDIO_KEY")

# Connect to the Label Studio API and check the connection
ls = Client(url=LABEL_STUDIO_URL, api_key=API_KEY)
ls.check_connection()

project = ls.get_project(22)

def read_and_encode(filename):
    with open(filename, "rb") as f:
        encoded_string = base64.b64encode(f.read())
    return encoded_string.decode("utf-8")

next_data = pd.read_csv(ARGS.data_path)

start = 0
finish = 100
split_data = [] 

for i in range(round(len(next_data)/100)):

    split = next_data[start:finish]
    split_data.append(split)

    start = finish
    finish = finish+100


for data_sample in tqdm(split_data):
    
    
    for idx, image in data_sample.iterrows():
        image = read_and_encode(image["path"])

        task = {
                    "data": {
                        "image": f"data:image/jpeg;base64,{image}",
                        "idx": str(idx),
                        "lat": data_sample["lat"][idx],
                        "long": data_sample["long"][idx],
                        "image_type": data_sample["image_type"][idx].astype(str),
                        "path": data_sample["path"][idx]
                        }
                    }

        project.import_tasks(
            [task]
        )

    time.sleep(60)

print('==== FINISHED ====')