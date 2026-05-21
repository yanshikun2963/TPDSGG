#!/bin/bash
# Setup symlinks to existing VG data and pretrained detector on AutoDL
# Run this ONCE after cloning the repo

DATA_SRC="/root/autodl-tmp/penet-main/datasets/vg"
# Try multiple possible paths for detector checkpoint (case sensitivity on Linux)
CKPT_CANDIDATES=(
    "/root/autodl-tmp/penet-main/Datasets/VG/model_final.pth"
    "/root/autodl-tmp/penet-main/datasets/vg/model_final.pth"
    "/root/autodl-tmp/penet-main/datasets/VG/model_final.pth"
)

# Create directories
mkdir -p datasets/vg
mkdir -p checkpoints/pretrained_faster_rcnn

# Symlink VG dataset files
for f in VG-SGG-with-attri.h5 VG-SGG-dicts-with-attri.json VG_100K image_data.json Category_Type_Info.json; do
    if [ ! -e "datasets/vg/$f" ]; then
        ln -s "$DATA_SRC/$f" "datasets/vg/$f"
        echo "Linked: datasets/vg/$f"
    fi
done

# Symlink GloVe files
for f in glove.6B.200d.txt glove.6B.300d.txt glove.6B.300d.pt glove.6B.50d.txt glove.6B.100d.txt; do
    if [ -e "$DATA_SRC/$f" ] && [ ! -e "datasets/vg/$f" ]; then
        ln -s "$DATA_SRC/$f" "datasets/vg/$f"
        echo "Linked: datasets/vg/$f"
    fi
done

# Symlink pretrained detector (try multiple candidate paths)
if [ ! -e "checkpoints/pretrained_faster_rcnn/model_final.pth" ]; then
    FOUND=0
    for CKPT_SRC in "${CKPT_CANDIDATES[@]}"; do
        if [ -e "$CKPT_SRC" ]; then
            ln -s "$CKPT_SRC" "checkpoints/pretrained_faster_rcnn/model_final.pth"
            echo "Linked detector: $CKPT_SRC"
            FOUND=1
            break
        fi
    done
    if [ $FOUND -eq 0 ]; then
        echo "ERROR: Cannot find pretrained detector model_final.pth!"
        echo "Tried: ${CKPT_CANDIDATES[*]}"
        exit 1
    fi
fi

# Create output directory
mkdir -p output

echo "Setup complete! Data symlinks created."
echo "Verify with: ls -la datasets/vg/ && ls -la checkpoints/pretrained_faster_rcnn/"
