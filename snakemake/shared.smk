configfile: "shared.yaml"


from datetime import date
import pandas as pd

TODAY = date.today().isoformat()
DATE = config.get("date", "no_date")
IN_DATE = config.get("in_date", "no_date")
TEST = config["test"]

RAW = config["data"]["raw"]
PROCESSED = config["data"]["processed"] if not TEST else config["out"]["tests"]
TEMP = config["data"]["temp"]
TEST_DATA = config["data"]["tests"]
META = config["data"]["meta"]
REMOTE = config["remote"] if not config["test"] else config["out"]["tests"]
OUT = config["out"]["root"] if not config["test"] else config["out"]["tests"]
BIN = config["bin"]
SRC = config["src"]
