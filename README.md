# Graduation Thesis

The implementation for **"[Learning Robust and Generalizable Representations for Continual Learning]"**

------

## ▶️ Usage

### **1. Create env and install requirements**

If you are using conda environment

```bash
conda create -n LRGR python=3.11
conda activate LRGR
pip install -r requirements.txt
```

Or if you are using uv instead

```bash
uv venv LRGR
source LRGR/bin/activate
uv pip install -r requirement.txt
```

### **2. Run the example training script**

```bash
bash LRGR.sh
```

### Project structure overview

```bash
LEAR/
├── backbone/                 # Pre-trained backbone models
│   ├── LEAR.py               # LEAR backbone implementation
│   └── ...
├── datasets/                 # Dataset loaders
|   ├── init.py               # Modify domain sequence                
│   └── ...
├── models/                   # CIL Method implementations
│   └── LEAR.py               # LEAR method with HRM implementation
├── utils/                    # Helper tools
|   ├── train_domain.py       # Training scripts                
│   └── ...
├── main_domain.py            # Main entry
├── LEAR.sh
└── README.md
```

------

## 🙏 Acknowledgement

This code is based on these papers below:

```bibtex
@inproceedings{lear,
  title={Learning Expandable and Adaptable Representations for Continual Learning},
  author={Yu, Ruilong and Liu, Mingyan and Ye, Fei and Bors, Adrian G and Hu, Rongyao and others},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}

@inproceedings{hrm_pet,
title={Hybrid Re-matching for Continual Learning with Parameter-Efficient Tuning},
author={Weicheng Wang and Guoli Jia and Xialei Liu and Liang Lin and Jufeng Yang},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2026},
}
```
