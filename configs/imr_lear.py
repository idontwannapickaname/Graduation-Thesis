import argparse


def get_args_parser(subparsers):
    subparsers.add_argument('--batch-size', default=32, type=int, help='Batch size per device')
    subparsers.add_argument('--epochs', default=5, type=int, help='Number of epochs per task')
    subparsers.add_argument('--lr', type=float, default=0.03, help='Learning rate')
    subparsers.add_argument('--seed', default=42, type=int, help='Random seed')
    subparsers.add_argument('--num_workers', default=4, type=int, help='DataLoader workers')
    subparsers.add_argument('--device', default='cuda', type=str, help='Training device')
    subparsers.add_argument('--data-path', default='./datasets', type=str, help='Root data path')
    subparsers.add_argument('--output_dir', default='./output/imr_lear', type=str, help='Output folder')

    # LEAR / Mammoth args
    subparsers.add_argument('--dataset', default='seq-imagenet-r', type=str, help='LEAR dataset name')
    subparsers.add_argument('--model_name', default='LEAR', type=str, help='LEAR model name')
    subparsers.add_argument('--backbone', default='lear', type=str, help='LEAR backbone name')
    subparsers.add_argument('--base_path', default='./data/', type=str, help='Base path used by LEAR')

    # Keep compatibility with existing main dispatch logic
    subparsers.add_argument('--train_inference_task_only', action='store_true')
