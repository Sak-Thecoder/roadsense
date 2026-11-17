"""Local desktop labeling UI. Saves each decision to the index CSV."""
import argparse
import os
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import messagebox
from PIL import Image, ImageOps, ImageTk
from roadsense.pipeline import FIELDS, LABELS, read_csv, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Index CSV to update in place')
    args = parser.parse_args()
    path = Path(args.input).resolve()
    rows = read_csv(path)
    if not rows:
        parser.error('Index is empty')
    root = tk.Tk()
    root.title('RoadSense manual quality review')
    index = [next((i for i, r in enumerate(rows) if not r.get('quality_label')), 0)]
    title = tk.Label(root, wraplength=900); title.pack()
    preview = tk.Label(root); preview.pack()
    status = tk.Label(root); status.pack()
    def render():
        row = rows[index[0]]
        title.config(text=f"{index[0]+1}/{len(rows)}  {row['original_path']}\nSource: {row['source']}   Damage annotations: {row.get('damage_types') or 'none recorded'}")
        try:
            with Image.open(row['original_path']) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((900, 600))
                preview.image = ImageTk.PhotoImage(im)
                preview.config(image=preview.image, text='')
        except OSError as exc:
            preview.config(image='', text=f'Cannot open image: {exc}')
        done = sum(bool(r.get('quality_label')) for r in rows)
        status.config(text=f"Reviewed: {done}/{len(rows)}   Current: {row.get('quality_label') or 'Unlabeled'}")
    def save(label):
        old = rows[index[0]].get('quality_label', '')
        rows[index[0]]['quality_label'] = label
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, suffix='.csv') as f:
                tmp = f.name
            # Preserve columns added by a user or later pipeline stage.
            write_csv(tmp, rows, list(rows[0].keys()))
            os.replace(tmp, path)
        except OSError as exc:
            rows[index[0]]['quality_label'] = old
            if tmp and os.path.exists(tmp): os.unlink(tmp)
            messagebox.showerror('Save failed', str(exc)); return
        index[0] = min(index[0] + 1, len(rows) - 1)
        render()
    buttons = tk.Frame(root); buttons.pack()
    for label in ('Good', 'Moderate', 'Poor', 'Uncertain'):
        tk.Button(buttons, text=label, command=lambda label=label: save(label)).pack(side='left')
    def move(n):
        index[0] = max(0, min(len(rows)-1, index[0]+n)); render()
    tk.Button(buttons, text='Previous', command=lambda: move(-1)).pack(side='left')
    tk.Button(buttons, text='Next', command=lambda: move(1)).pack(side='left')
    render(); root.mainloop()

if __name__ == '__main__':
    main()
