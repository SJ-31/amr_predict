configfile: "shared.yaml"


from datetime import date
import pandas as pd

TODAY = date.today().isoformat()
DATE = config.get("date")

RAW = config["data"]["raw"]
PROCESSED = config["data"]["processed"]
TEMP = config["data"]["temp"]
REMOTE = config["remote"] if not config["test"] else config["data"]["temp"]
OUT = config["out"]["root"] if not config["test"] else config["out"]["tests"]
BIN = config["bin"]
SRC = config["src"]
