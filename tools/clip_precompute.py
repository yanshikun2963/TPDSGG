#!/usr/bin/env python
"""
CLIP/OpenCLIP Embedding Precomputation for CAPE-SGG

Usage:
    python tools/clip_precompute.py --output_dir ./datasets/vg/
    
    # With CLIP ViT-B/32 fallback:
    python tools/clip_precompute.py --use_openai_clip --output_dir ./datasets/vg/
"""

import os
import sys
import json
import argparse
import torch

# VG150 Object Classes (151 classes including background)
VG150_OBJ_CLASSES = [
    '__background__', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket',
    'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat',
    'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building',
    'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock',
    'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door',
    'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger',
    'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass',
    'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet',
    'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady',
    'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man',
    'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number',
    'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone',
    'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole',
    'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen',
    'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign',
    'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock',
    'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile',
    'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree',
    'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle',
    'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman',
    'zebra'
]

# VG150 Predicate Classes (51 classes including background)
VG150_PRED_CLASSES = [
    '__background__', 'above', 'across', 'against', 'along', 'and', 'at',
    'attached to', 'behind', 'belonging to', 'between', 'carrying',
    'covered in', 'covering', 'eating', 'flying in', 'for', 'from',
    'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of',
    'laying on', 'looking at', 'lying on', 'made of', 'mounted on',
    'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on',
    'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on',
    'to', 'under', 'using', 'walking in', 'walking on', 'watching',
    'wearing', 'wears', 'with'
]


def encode_with_open_clip(model_name='ViT-L-14', pretrained='datacomp_xl_s13b_b90k',
                          obj_classes=None, pred_classes=None):
    """Encode using OpenCLIP (preferred method)."""
    if obj_classes is None:
        obj_classes = VG150_OBJ_CLASSES
    if pred_classes is None:
        pred_classes = VG150_PRED_CLASSES
    try:
        import open_clip
    except ImportError:
        print("[ERROR] open_clip not installed. Install with: pip install open-clip-torch")
        return None, None, None, None
    
    print(f"Loading OpenCLIP model: {model_name} ({pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Encoding {len(obj_classes)} object classes...")
    obj_prompts = [f"a photo of a {name}" for name in obj_classes]
    obj_tokens = tokenizer(obj_prompts)
    with torch.no_grad():
        obj_embeds = model.encode_text(obj_tokens.to(device))
        obj_embeds = obj_embeds / obj_embeds.norm(dim=-1, keepdim=True)
    
    print(f"Encoding {len(pred_classes)} predicate classes...")
    pred_prompts = [f"something {name} something" for name in pred_classes]
    pred_tokens = tokenizer(pred_prompts)
    with torch.no_grad():
        pred_embeds = model.encode_text(pred_tokens.to(device))
        pred_embeds = pred_embeds / pred_embeds.norm(dim=-1, keepdim=True)
    
    embed_dim = obj_embeds.shape[1]
    model_id = f"openclip-{model_name}-{pretrained}"
    
    return obj_embeds.cpu().float(), pred_embeds.cpu().float(), embed_dim, model_id


def encode_with_clip(model_name='ViT-B/32', obj_classes=None, pred_classes=None):
    """Encode using OpenAI CLIP (fallback method)."""
    if obj_classes is None:
        obj_classes = VG150_OBJ_CLASSES
    if pred_classes is None:
        pred_classes = VG150_PRED_CLASSES
    try:
        import clip
    except ImportError:
        print("[ERROR] clip not installed. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None, None, None, None
    
    print(f"Loading CLIP model: {model_name}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = clip.load(model_name, device=device)
    model.eval()
    
    print(f"Encoding {len(obj_classes)} object classes...")
    obj_prompts = [f"a photo of a {name}" for name in obj_classes]
    obj_tokens = clip.tokenize(obj_prompts, truncate=True).to(device)
    with torch.no_grad():
        obj_embeds = model.encode_text(obj_tokens)
        obj_embeds = obj_embeds / obj_embeds.norm(dim=-1, keepdim=True)
    
    print(f"Encoding {len(pred_classes)} predicate classes...")
    pred_prompts = [f"something {name} something" for name in pred_classes]
    pred_tokens = clip.tokenize(pred_prompts, truncate=True).to(device)
    with torch.no_grad():
        pred_embeds = model.encode_text(pred_tokens)
        pred_embeds = pred_embeds / pred_embeds.norm(dim=-1, keepdim=True)
    
    embed_dim = obj_embeds.shape[1]
    model_id = f"clip-{model_name.replace('/', '-')}"
    
    return obj_embeds.cpu().float(), pred_embeds.cpu().float(), embed_dim, model_id


def main():
    parser = argparse.ArgumentParser(description='Precompute CLIP embeddings for CAPE-SGG')
    parser.add_argument('--output_dir', type=str, default='./datasets/vg/',
                        help='Output directory for embeddings file')
    parser.add_argument('--model', type=str, default='ViT-L-14',
                        help='Model name (OpenCLIP format)')
    parser.add_argument('--pretrained', type=str, default='datacomp_xl_s13b_b90k',
                        help='Pretrained weights (OpenCLIP format)')
    parser.add_argument('--use_openai_clip', action='store_true',
                        help='Use OpenAI CLIP instead of OpenCLIP')
    parser.add_argument('--dict_file', type=str, default=None,
                        help='Path to VG-SGG-dicts-with-attri.json (uses actual dataset class order)')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine class names: from dict_file if available, else hardcoded
    obj_classes, pred_classes = VG150_OBJ_CLASSES, VG150_PRED_CLASSES
    if args.dict_file and os.path.exists(args.dict_file):
        print(f"Loading class names from: {args.dict_file}")
        with open(args.dict_file, 'r') as f:
            vg_dict = json.load(f)
        if 'idx_to_label' in vg_dict:
            max_obj_idx = max(int(k) for k in vg_dict['idx_to_label'].keys())
            obj_classes = ['__background__'] + [vg_dict['idx_to_label'][str(i)] for i in range(1, max_obj_idx + 1)]
        if 'idx_to_predicate' in vg_dict:
            max_pred_idx = max(int(k) for k in vg_dict['idx_to_predicate'].keys())
            pred_classes = ['__background__'] + [vg_dict['idx_to_predicate'][str(i)] for i in range(1, max_pred_idx + 1)]
        print(f"  Objects: {len(obj_classes)}, Predicates: {len(pred_classes)}")
    else:
        if args.dict_file:
            print(f"[WARNING] Dict file not found: {args.dict_file}. Using hardcoded VG150 names.")
        # Also try auto-detect
        for candidate in ['./datasets/vg/VG-SGG-dicts-with-attri.json',
                          '../datasets/vg/VG-SGG-dicts-with-attri.json']:
            if os.path.exists(candidate):
                print(f"Auto-detected dict file: {candidate}")
                with open(candidate, 'r') as f:
                    vg_dict = json.load(f)
                if 'idx_to_label' in vg_dict:
                    max_obj_idx = max(int(k) for k in vg_dict['idx_to_label'].keys())
                    obj_classes = ['__background__'] + [vg_dict['idx_to_label'][str(i)] for i in range(1, max_obj_idx + 1)]
                if 'idx_to_predicate' in vg_dict:
                    max_pred_idx = max(int(k) for k in vg_dict['idx_to_predicate'].keys())
                    pred_classes = ['__background__'] + [vg_dict['idx_to_predicate'][str(i)] for i in range(1, max_pred_idx + 1)]
                print(f"  Objects: {len(obj_classes)}, Predicates: {len(pred_classes)}")
                break
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.use_openai_clip:
        obj_embeds, pred_embeds, embed_dim, model_id = encode_with_clip(
            args.model, obj_classes=obj_classes, pred_classes=pred_classes)
    else:
        result = encode_with_open_clip(args.model, args.pretrained,
                                        obj_classes=obj_classes, pred_classes=pred_classes)
        if result[0] is None:
            print("Falling back to OpenAI CLIP ViT-B/32...")
            obj_embeds, pred_embeds, embed_dim, model_id = encode_with_clip(
                'ViT-B/32', obj_classes=obj_classes, pred_classes=pred_classes)
        else:
            obj_embeds, pred_embeds, embed_dim, model_id = result
    
    if obj_embeds is None:
        print("[FATAL] No CLIP library available. Please install open-clip-torch or clip.")
        sys.exit(1)
    
    output_path = os.path.join(args.output_dir, 'clip_embeddings.pt')
    torch.save({
        'obj_embeddings': obj_embeds,
        'pred_embeddings': pred_embeds,
        'embed_dim': embed_dim,
        'model_name': model_id,
        'obj_classes': obj_classes,
        'pred_classes': pred_classes,
    }, output_path)
    
    print(f"\n=== CLIP Embeddings Saved ===")
    print(f"  Model: {model_id}")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Object classes: {obj_embeds.shape}")
    print(f"  Predicate classes: {pred_embeds.shape}")
    print(f"  Output: {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == '__main__':
    main()
