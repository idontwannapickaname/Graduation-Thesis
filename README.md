# LEAR

The implementation for **"[Learning Expandable and Adaptable Representations for Continual Learning]"**

------

## ▶️ Usage

### **1. Create env and install requirements**

If you are using conda environment

```bash
conda create -n LEAR python=3.11
conda activate LEAR
pip install -r requirements.txt
```

Or if you are using uv instead

```bash
uv venv LEAR
source LEAR/bin/activate
uv pip install -r requirement.txt
```

### **2. Run the example training script**

```bash
bash LEAR.sh
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
├── models/                   # CL Method implementations
│   └── LEAR.py               # LEAR method implementation
├── utils/                    # Helper tools
|   ├── train_domain.py       # Training scripts                
│   └── ...
├── main_domain.py            # Main entry
├── LEAR.sh
└── README.md
```

------

## 🙏 Acknowledgement

This code is based on this papers below:

```bibtex
@inproceedings{yulearning,
  title={Learning Expandable and Adaptable Representations for Continual Learning},
  author={Yu, Ruilong and Liu, Mingyan and Ye, Fei and Bors, Adrian G and Hu, Rongyao and others},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```
