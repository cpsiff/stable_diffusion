import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_colwidth = 150

INDEX_FILE = "index2.yaml"

# load metrics in from yaml file
with open(INDEX_FILE, 'r') as f:
    index = []
    f_str = f.read()
    for chunk in f_str.split("\n\n"):
        if chunk.rstrip() != "":
            chunk_dict = yaml.safe_load(chunk)
            index.append(chunk_dict[list(chunk_dict.keys())[0]])

df = pd.DataFrame(index)

df['is_nsfw'] = df['is_nsfw'].astype("string")

df = df.loc[df['is_nsfw'] == "[False]"]

# df["parameters"] = df["guidance_scale"].astype(str) + "-" + df["prompt"] + "-" + df["seed"].astype(str) + "-" + df["strength"].astype(str)
# df = df.drop(columns=["prompt", "guidance_scale", "seed", "strength"])

# df = df.groupby("parameters").mean().sort_values(by="SSIM", ascending=False)

# print(df)

sns.lineplot(df, x="strength", y="SSIM")
plt.show()