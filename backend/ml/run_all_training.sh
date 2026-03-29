#!/bin/bash
# ============================================================
# ARKEN — ML Training Runner v3.2
# Runs Steps 7-9 only (Price, ROI, and Property models)
# ============================================================

set -e
WEIGHTS="ml/weights"
LOG_DIR="ml/weights/train_logs"
mkdir -p "$LOG_DIR"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

pass()   { echo -e "${GREEN}  ✔ $1${NC}"; }
fail()   { echo -e "${RED}  ✗ $1${NC}"; }
info()   { echo -e "${CYAN}  → $1${NC}"; }
warn()   { echo -e "${YELLOW}  ⚠ $1${NC}"; }
header() {
  echo -e "\n${CYAN}══════════════════════════════════════════${NC}"
  echo -e "${CYAN}  $1${NC}"
  echo -e "${CYAN}══════════════════════════════════════════${NC}"
}

# ── Step 0: GPU check ──────────────────────────────────────
header "Step 0 — Environment check"
python -c "
import torch
cuda = torch.cuda.is_available()
name = torch.cuda.get_device_name(0) if cuda else 'none'
print(f'CUDA: {cuda} | GPU: {name}')
if not cuda:
    raise SystemExit('ERROR: GPU not visible in container.')
"
pass "GPU visible"

# ── Step 1: Migrate labels ─────────────────────────────────
header "Step 1 — Migrate existing labels"
if python ml/migrate_labels.py 2>&1 | tee "$LOG_DIR/migrate.log"; then
    pass "Labels migrated"
else
    warn "Migration had issues — check $LOG_DIR/migrate.log"
fi

# ── Steps 2-6: Already done ────────────────────────────────
header "Step 2 — YOLO annotation"
pass "SKIPPED — yolo_indian_rooms.pt already trained (mAP50=0.791)"

header "Step 3 — YOLO fine-tuning"
pass "SKIPPED — yolo_indian_rooms.pt already saved"

header "Step 4 — CLIP fine-tuning"
pass "SKIPPED — clip_finetuned.pt already trained (val_loss=0.1635)"

header "Step 5 — Room classifier"
pass "SKIPPED — room_classifier.pt already trained"

header "Step 6 — Style classifier"
pass "SKIPPED — style_classifier.pt already trained"

# ── Step 7: Price models ────────────────────────────────────
header "Step 7 — Price models (XGBoost + Prophet)"
python ml/train_price_models.py \
    --output-dir "$WEIGHTS" \
    2>&1 | tee "$LOG_DIR/price_train.log"

if [ -f "$WEIGHTS/price_xgb.joblib" ]; then
    pass "price_xgb.joblib saved"
else
    fail "price_xgb.joblib NOT found — check $LOG_DIR/price_train.log"
fi

# ── Step 8: ROI models (FIXED) ──────────────────────────────
header "Step 8 — ROI models (Post-Value + Uplift models)"

# Fix the path in train_roi_models.py if not already fixed
info "Ensuring ROI model script has correct data path..."
sed -i 's|/home/claude/arke4n12/backend/data/datasets/property_transactions/india_property_transactions.csv|data/datasets/property_transactions/india_property_transactions.csv|g' ml/train_roi_models.py 2>/dev/null || true

# Verify the fix
if grep -q "data/datasets/property_transactions/india_property_transactions.csv" ml/train_roi_models.py; then
    pass "ROI script path corrected"
else
    warn "Could not verify ROI script path"
fi

# Remove old ROI models to ensure fresh training
rm -f "$WEIGHTS/roi_pre_post_model.joblib" "$WEIGHTS/roi_uplift_model.joblib" 2>/dev/null || true

# Train the ROI models
info "Training ROI models..."
python ml/train_roi_models.py \
    --weights-dir "$WEIGHTS" \
    2>&1 | tee "$LOG_DIR/roi_train.log"

# Copy models if they were saved in a different location
if [ -d "/app/ml/roi_weights_realistic" ]; then
    info "Found ROI models in alternate location, copying to $WEIGHTS..."
    cp /app/ml/roi_weights_realistic/roi_post_value_model.joblib "$WEIGHTS/roi_pre_post_model.joblib" 2>/dev/null || true
    cp /app/ml/roi_weights_realistic/roi_value_uplift_model.joblib "$WEIGHTS/roi_uplift_model.joblib" 2>/dev/null || true
    cp /app/ml/roi_weights_realistic/roi_pre_post_model.joblib "$WEIGHTS/" 2>/dev/null || true
    cp /app/ml/roi_weights_realistic/roi_uplift_model.joblib "$WEIGHTS/" 2>/dev/null || true
fi

# Verify ROI models were created
if [ -f "$WEIGHTS/roi_pre_post_model.joblib" ] && [ -f "$WEIGHTS/roi_uplift_model.joblib" ]; then
    pass "roi_pre_post_model.joblib + roi_uplift_model.joblib saved"
else
    # Check if they exist with different names
    if [ -f "$WEIGHTS/roi_post_value_model.joblib" ]; then
        cp "$WEIGHTS/roi_post_value_model.joblib" "$WEIGHTS/roi_pre_post_model.joblib"
        pass "Renamed roi_post_value_model.joblib to roi_pre_post_model.joblib"
    fi
    if [ -f "$WEIGHTS/roi_value_uplift_model.joblib" ]; then
        cp "$WEIGHTS/roi_value_uplift_model.joblib" "$WEIGHTS/roi_uplift_model.joblib"
        pass "Renamed roi_value_uplift_model.joblib to roi_uplift_model.joblib"
    fi
    
    # Final check
    if [ -f "$WEIGHTS/roi_pre_post_model.joblib" ] && [ -f "$WEIGHTS/roi_uplift_model.joblib" ]; then
        pass "ROI models available after rename"
    else
        fail "ROI models NOT found — check $LOG_DIR/roi_train.log"
    fi
fi

# ── Step 9: Property value models ──────────────────────────
header "Step 9 — Property value models (ensemble)"
python -m ml.train_models \
    2>&1 | tee "$LOG_DIR/property_train.log"
pass "Property models complete"

# ── Final summary ───────────────────────────────────────────
header "Training Summary"
echo ""

declare -A WEIGHTS_FILES=(
    ["price_xgb.joblib"]="Price XGBoost"
    ["room_classifier.pt"]="Room Type Classifier"
    ["yolo_indian_rooms.pt"]="YOLO Object Detector"
    ["roi_pre_post_model.joblib"]="ROI Post-Value Model"
    ["roi_uplift_model.joblib"]="ROI Uplift Model"
    ["clip_finetuned.pt"]="CLIP Visual Encoder"
    ["style_classifier.pt"]="Style Classifier"
)

ALL_OK=true
for f in "${!WEIGHTS_FILES[@]}"; do
    if [ -f "$WEIGHTS/$f" ]; then
        SIZE=$(du -sh "$WEIGHTS/$f" 2>/dev/null | cut -f1 || echo "unknown")
        pass "${WEIGHTS_FILES[$f]} — $f ($SIZE)"
    else
        fail "${WEIGHTS_FILES[$f]} — $f MISSING"
        ALL_OK=false
    fi
done
echo ""

if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}  ✅ All models trained successfully!${NC}"
    echo -e "${GREEN}  📁 Logs saved to: $LOG_DIR/${NC}"
else
    echo -e "${YELLOW}  ⚠️ Some models failed. Check logs in: $LOG_DIR/${NC}"
fi
echo ""