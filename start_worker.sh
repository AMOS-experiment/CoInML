source .env

export PREFECT_WORK_DIR=$PWD
export PREFECT_LOCAL_STORAGE_PATH=$PWD/data/prefect-storage
prefect config set PREFECT_API_URL=$PREFECT_API_URL

prefect work-pool create sculpt-pool --type "process"
prefect work-pool set-concurrency-limit sculpt-pool $PREFECT_WORK_POOL_CONCURRENCY
prefect deploy --all

prefect worker start --pool sculpt-pool --limit $PREFECT_WORKER_LIMIT --with-healthcheck
