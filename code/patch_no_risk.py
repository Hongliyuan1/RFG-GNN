from pathlib import Path

target = Path("train1221_dtbe_no_risk_strict.py")
text = target.read_text(encoding="utf-8")

markers = {
    "risk_init": "self.risk_layer = nn.Linear(n_hidden, n_hidden)",
    "risk_compute": "risk_feature = self.risk_layer(x)",
    "risk_add": "x = x + 0.1 * F.relu(risk_feature)",
}

found = {name: 0 for name in markers}
new_lines = []

for line in text.splitlines(keepends=True):
    matched = False

    for name, marker in markers.items():
        if marker in line:
            found[name] += 1
            matched = True
            break

    if not matched:
        new_lines.append(line)

for name, count in found.items():
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one occurrence of {name}, found {count}."
        )

target.write_text("".join(new_lines), encoding="utf-8")

print("No-risk strict script patched successfully:", target)
print("Removed:", found)
