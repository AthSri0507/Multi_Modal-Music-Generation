# Scaling Plan

Generated: 2026-03-31T15:57:24.462215

## Baseline
- Rows loaded: 50000
- Rows after dedup: 47784

## Phase 2 (300k)
- Mode: single machine + local Spark
- Deliverables: processed dataset + splits + quality report
- Acceptance: pipeline succeeds and quality checks pass

## Phase 3 (900k)
- Mode: distributed Spark
- Recommendations: driver 4g, executor 8g, higher shuffle partitions
- Fallback: chunked processing if resources are constrained
