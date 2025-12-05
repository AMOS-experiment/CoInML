#!/bin/bash
set -e

# Create work pool (ignore if it already exists)
prefect work-pool create sculpt-pool --type "process" 2>/dev/null || true

# Update concurrency limit
prefect work-pool update sculpt-pool --concurrency-limit ${PREFECT_WORK_POOL_CONCURRENCY:-4}

# Deploy flows
prefect deploy --all

# Start worker
exec prefect worker start -p sculpt-pool --limit ${PREFECT_WORKER_LIMIT:-4} --with-healthcheck
