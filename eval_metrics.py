import yaml
import pandas as pd

INDEX_FILE = "/home/carter/repos/stable-diffusion/test/output/20221201-205305/index.yaml"

# load metrics in from yaml file
with open(INDEX_FILE) as f:
    index = yaml.safe_load(f)

pd.DataFrame()