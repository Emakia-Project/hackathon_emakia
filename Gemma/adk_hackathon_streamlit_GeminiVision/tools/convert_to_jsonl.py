"""
convert_to_jsonl.py
────────────────────
Converts a local CoreML CSV file to Gemini fine-tuning JSONL format.

Input CSV expected columns:  text, label
Output JSONL format required by Vertex AI Gemini tuning:
  {"contents": [
      {"role": "user",  "parts": [{"text": "tweet text here"}]},
      {"role": "model", "parts": [{"text": "Harassment"}]}
  ]}

Usage:
    python convert_to_jsonl.py
    python convert_to_jsonl.py --input my_data.csv --output training.jsonl --text_col text --label_col label
"""

import json
import csv
import argparse


def convert(input_csv: str, output_jsonl: str, text_col: str, label_col: str, remove_skipped: bool = False):
    skipped = 0
    written = 0
    skipped_rows = []

    with open(input_csv, "r", encoding="utf-8") as csv_file, \
         open(output_jsonl, "w", encoding="utf-8") as jsonl_file:

        reader = csv.DictReader(csv_file)

        # Strip whitespace from column headers (handles ' label ' → 'label')
        reader.fieldnames = [f.strip() for f in reader.fieldnames]

        # Also strip whitespace from values when reading
        # Validate columns exist
        if text_col not in reader.fieldnames or label_col not in reader.fieldnames:
            raise ValueError(
                f"Columns '{text_col}' and/or '{label_col}' not found in CSV.\n"
                f"Available columns: {reader.fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):  # start=2 accounts for header
            # Strip whitespace from keys too (matches stripped fieldnames)
            row   = {k.strip(): v for k, v in row.items()}
            text  = row[text_col].strip()
            label = row[label_col].strip()

            # Skip rows with missing text or label
            if not text or not label:
                skipped += 1
                skipped_rows.append({
                    "csv_row": row_num,
                    "text":    text  if text  else "⚠️  EMPTY",
                    "label":   label if label else "⚠️  EMPTY",
                    "raw":     dict(row),
                })
                continue

            record = {
                "contents": [
                    {"role": "user",  "parts": [{"text": text}]},
                    {"role": "model", "parts": [{"text": label}]}
                ]
            }
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"✅ Done!")
    print(f"   Written : {written} rows  →  {output_jsonl}")
    if skipped:
        print(f"   Skipped : {skipped} rows (missing text or label)")
        print(f"\n{'─'*60}")
        print(f"⚠️  SKIPPED ROWS — review before deciding to remove:")
        print(f"{'─'*60}")
        for s in skipped_rows:
            print(f"  Row {s['csv_row']:>6} │ text: {s['text'][:60]!r:<62} │ label: {s['label']}")
        print(f"{'─'*60}")
        print(f"  To keep them:   edit the CSV and fill in the missing values")
        if remove_skipped:
            # Write a cleaned CSV with skipped rows removed
            import os
            # Write cleaned CSV next to the output JSONL, not next to the input
            base = os.path.splitext(os.path.basename(input_csv))[0]
            out_dir = os.path.dirname(os.path.abspath(output_jsonl))
            cleaned_csv = os.path.join(out_dir, f"{base}_cleaned.csv")
            skip_rows = {s["csv_row"] for s in skipped_rows}

            with open(input_csv, "r", encoding="utf-8") as src,                  open(cleaned_csv, "w", encoding="utf-8", newline="") as dst:
                reader2 = csv.reader(src)
                writer  = csv.writer(dst)
                for i, row in enumerate(reader2, start=1):
                    if i not in skip_rows:   # always keep header (row 1) + non-skipped rows
                        writer.writerow(row)

            print(f"\n🗑️  Cleaned CSV written → {cleaned_csv}")
            print(f"   Removed {len(skip_rows)} empty-text rows.")
            print(f"   Original : {written + skipped} rows")
            print(f"   Cleaned  : {written} rows")
        else:
            print(f"  To remove them: re-run with --remove_skipped flag")

    # Vertex AI recommends at least 100 examples per label
    print(f"\n📋 Checking label distribution...")
    label_counts: dict = {}
    with open(output_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            lbl = record["contents"][1]["parts"][0]["text"]
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    for lbl, count in sorted(label_counts.items()):
        status = "✅" if count >= 100 else "⚠️  (Vertex AI recommends at least 100)"
        print(f"   {lbl}: {count} examples  {status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CoreML CSV to Gemini JSONL")
    parser.add_argument("--input",     default="coreml_training.csv", help="Path to input CSV file")
    parser.add_argument("--output",    default="gemini_training.jsonl", help="Path to output JSONL file")
    parser.add_argument("--text_col",  default="text",  help="Name of the text column in CSV")
    parser.add_argument("--label_col",      default="label", help="Name of the label column in CSV")
    parser.add_argument("--remove_skipped", action="store_true",
                        help="Also write a cleaned CSV with the skipped rows removed")
    args = parser.parse_args()

    convert(args.input, args.output, args.text_col, args.label_col, args.remove_skipped)
