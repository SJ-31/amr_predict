configfile: "shared.yaml"


from datetime import date
import pandas as pd

TODAY = date.today().isoformat()
DATE = config.get("date")

RAW = config["data"]["raw"]
PROCESSED = config["data"]["processed"]
REMOTE = config["remote"] if not config["test"] else config["data"]["temp"]
OUT = config["root"]["root"] if not config["test"] else config["out"]["tests"]
BIN = config["bin"]
