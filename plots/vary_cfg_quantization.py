import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('seaborn')

pd.options.display.max_colwidth = 150

INDEX_FILE = "vary_cfg.yaml"

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

# zero = pd.DataFrame([{
#     "L2": 0,
#     "PSNR": 1.0,
#     "SSIM": 0.898,
#     "guidance_scale": 10,
#     "is_nsfw": [False],
#     "negative_prompt": "GIF, quantized, posterized",
#     "prompt": "shot on iPhone, 4K, high quality",
#     "seed": 100,
#     "strength": 0.0
# }]*50)

# df = pd.concat([df, zero], ignore_index=True)

fig, ax = plt.subplots()
fig.set_size_inches(6, 4)
sns.lineplot(df, ax=ax, x="guidance_scale", y="SSIM")

ax.set_xlabel("Guidance Scale", size=12)
ax.set_ylabel("Average SSIM", size=12)
# ax.set_title("Effect of Guidance Scale on SSIM - Color Quantization \n Diffusion strength fixed at 0.3")
ax.set_ylim(0, 1.0)
ax.plot([0.0, 30], [0.898, 0.898], ls="--", c="gray") # doing nothing
ax.plot([0.0, 30], [0.903, 0.903], ls="--", c="green") # deepdeband

plt.tight_layout()
plt.savefig("vary_cfg_quant.pdf", dpi=500)
plt.show()