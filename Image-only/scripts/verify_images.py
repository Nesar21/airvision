#!/usr/bin/env python3
import pandas as pd
import os
import argparse

def main(args):
    df = pd.read_csv(args.csv)

    df["exists"] = df["image_path"].apply(lambda p: os.path.exists(p))

    missing = df[~df["exists"]]
    present = df[df["exists"]]

    print("Total rows:", len(df))
    print("Present images:", len(present))
    print("Missing images:", len(missing))

    if len(missing) > 0:
        print("Writing missing_images.csv")
        missing.to_csv("missing_images.csv", index=False)

    print("Writing valid_images.csv")
    present.to_csv("valid_images.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="path to master_v1.csv")
    main(parser.parse_args())
