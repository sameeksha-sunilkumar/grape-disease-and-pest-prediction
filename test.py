import os

train_dir = r"C:\Users\DELL\Desktop\projects\grape disease detection\fruit_data\train"

total = 0
for cls in os.listdir(train_dir):
    n = len(os.listdir(os.path.join(train_dir, cls)))
    print(f"{cls}: {n}")
    total += n

print("Total:", total)
