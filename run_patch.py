import re

with open("src/train.py", "r") as f:
    text = f.read()

target = "for epoch in tqdm(range(1, args.epochs + 1)):"

replacement = "for epoch in tqdm(range(start_epoch, args.epochs + 1)):"

new_text = text.replace(target, replacement)
with open("src/train.py", "w") as f:
    f.write(new_text)

print("Done")
