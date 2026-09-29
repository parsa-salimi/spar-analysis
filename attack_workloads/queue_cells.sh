# Proposed cells for scripts/rebuttal_collection/multinode_queue.sh (S16 phase).
# Copy _mn_attack.py, train_diloco.py and train_ps_lopsided.py into workloads/.
# Every name contains "advtrain": routes to ml_training, stays out of the honest pool.
# Both need a full 1B replica + optimizer state per GPU -> 80 GB cards (need_gb 60).

# ── DiLoCo: rare cross-node synchronisation ─────────────────────────────────
c=s16_advtrain_diloco_h100;          need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco.py --size 1b --inner-steps 100 --outer-quant none --schedule burst  --duration "$DUR"
c=s16_advtrain_diloco_h500;          need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco.py --size 1b --inner-steps 500 --outer-quant none --schedule burst  --duration "$DUR"
c=s16_advtrain_diloco_h100_int4;     need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco.py --size 1b --inner-steps 100 --outer-quant int4 --schedule burst  --duration "$DUR"
c=s16_advtrain_diloco_h100_spread;   need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco.py --size 1b --inner-steps 100 --outer-quant none --schedule spread --duration "$DUR"
c=s16_advtrain_diloco_h500_int4_spread; need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco.py --size 1b --inner-steps 500 --outer-quant int4 --schedule spread --duration "$DUR"

# ── Lopsided parameter server: compressed upload, dense download ────────────
c=s16_advtrain_pslop_c05;            need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_ps_lopsided.py --size 1b --topk-frac 0.05 --duration "$DUR"
c=s16_advtrain_pslop_c10;            need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_ps_lopsided.py --size 1b --topk-frac 0.10 --duration "$DUR"
c=s16_advtrain_pslop_c30;            need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_ps_lopsided.py --size 1b --topk-frac 0.30 --duration "$DUR"
