# ChemEAGLE

## Original Repository
- https://github.com/CYF2000127/ChemEagle

This repository is adapted from the original ChemEAGLE project.

## Optimizations in This Fork
- 1. Replaced `AzureOpenAI` with the standard `openai` framework for better compatibility across providers and OpenAI-compatible APIs.
- 2. In mainland China, users can obtain usable API access through API aggregation platforms, making setup easier.
- 3. Changed model/weight download behavior to store files directly in the project directory instead of default global cache paths, reducing the risk of system disk exhaustion.
- 4. Improved usability for mainland China network environments (including HuggingFace mirror support).


## Using the Code

Clone the repository:

```bash
git clone https://github.com/LHH90538/ChemEagle-0326
```


1. Create and activate a conda environment:

```bash
conda create -n chemeagle python=3.10
conda activate chemeagle
```

2. Install requirements:

```bash
cd ChemEagle
pip install -r requirements.txt
```

3. Download necessary model files and place them in the project root.
You can auto-download them by running:

If you are in mainland China and access to HuggingFace is restricted, run this first to switch to a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

```bash
python zip_downloader.py
python ckpt_downloader.py
```

Tesseract OCR is also required and should be installed from the Linux package manager:

```bash
sudo apt update
sudo apt install tesseract-ocr
```

Required files/directories include:
- `rxn.ckpt`
- `molnextr.pth`
- `moldet.ckpt`
- `corefdet.ckpt`
- `ner.ckpt`
- `cre_models_v0.1/`
- `biobert-large-cased/`

4. Set OpenAI API environment variables:

```bash
export apikey=your-api-key
export baseurl=xxxxx
```

5. Run extraction on a chemical graphic:

```python
from main import ChemEagle

image_path = "./examples/1.png"
results = ChemEagle(image_path)
print(results)
```
