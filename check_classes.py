from pathlib import Path

def scan(folder):
    classes = set()
    bad = 0
    for p in Path(folder).glob("*.txt"):
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                classes.add(int(parts[0]))
            except:
                bad += 1
    return sorted(classes), bad

train_classes, train_bad = scan("datasets/cars/labels/train")
val_classes, val_bad = scan("datasets/cars/labels/val")

print("TRAIN classes:", train_classes, "bad lines:", train_bad)
print("VAL classes:  ", val_classes, "bad lines:", val_bad)

all_classes = train_classes + [c for c in val_classes if c not in train_classes]
print("Max class id:", max(all_classes) if all_classes else "NONE")
