## Run

```bash
git clone https://github.com/Chal1ce/mandarin-fish-growth.git
cd mandarin-fish-growth
python -m pip install -r requirements.txt
```

```bash
python -m zipfile -e Mandarin_fish_processed_data_v1.zip data
python reproduce.py --data-dir data/Mandarin_fish_processed_data --output-dir outputs
```

## Dataset

https://doi.org/10.5281/zenodo.23390429

## Citation

```bibtex
@dataset{chen2026mandaringrowth,
  author = {Chen, Shengnan},
  title = {Processed datasets for cross-study growth prediction in mandarin fish (Siniperca chuatsi)},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23390429},
  url = {https://doi.org/10.5281/zenodo.23390429}
}
```

Code: https://github.com/Chal1ce/mandarin-fish-growth
