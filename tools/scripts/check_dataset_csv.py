import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(
    description="Checks a csv dataset, whether its images are present on the disk.")
parser.add_argument("--img_dir", type=str, required=True,
                    help="root directory for the images")
parser.add_argument("--csv_file", type=str, required=True,
                    help="csv file to check for validity.")
args = parser.parse_args()

df = pd.read_csv(args.csv_file)

total_missing = 0
for sub_pth in df["path"]:
    pth = os.path.join(args.img_dir, sub_pth)
    if not os.path.isfile(pth):
        total_missing += 1
        print(f"Missing file: {sub_pth}")

print("Check finished. Total missing files: {}/{}.".format(total_missing, len(df)))
