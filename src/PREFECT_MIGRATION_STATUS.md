# SCULPT Prefect Migration Documentation

## Overview

Migration of SCULPT (Supervised Clustering and Uncovering Latent Patterns with Training) from synchronous Dash callbacks to asynchronous Prefect workflow orchestration to prevent UI blocking during computationally intensive operations.

**Date Started**: October 10, 2025  
**Current Status**: UMAP integration in progress  
**Python Version**: 3.9.13  
**Platform**: macOS arm64  
**Prefect Version**: 3.4.23  
**Virtual Environment**: sculpt-test  

---

## Project Context

### What is SCULPT?
- Interactive ML platform for COLTRIMS (Cold Target Recoil Ion Momentum Spectroscopy) data analysis
- Web-based Dash application for molecular physics research
- Processes ~19,000 data points with complex ML operations
- Key features: UMAP dimensionality reduction, genetic programming, autoencoder training

### Why Prefect?
**Problem**: Long-running operations (genetic programming, autoencoder training, UMAP) block the UI, making the application unresponsive.

**Solution**: Prefect workflow orchestration provides:
- Non-blocking UI (tasks run in background)
- Real-time progress tracking
- Automatic caching of expensive computations
- Beautiful monitoring dashboard
- ML-optimized retry logic and error handling
- Academic research-friendly (simple setup, great for papers)

**Why Prefect over alternatives?**
- **vs Celery**: Much simpler setup, better ML support, no complex broker configuration
- **vs RQ**: More sophisticated workflow management, better progress tracking
- **vs Airflow**: More intuitive Python-first approach, no DAG complexity

---

## Architecture

### Before (Synchronous)
```
Dash Frontend ←→ Callbacks ←→ [PyTorch, sklearn, UMAP] 
                                (all blocking in same process)
```

### After (Asynchronous with Prefect)
```
Dash Frontend ←→ Prefect Flow Submission ←→ Prefect Server ←→ Background Workers
       ↓                                           ↓
  Responsive UI                            [PyTorch, sklearn, UMAP]
       ↓                                           ↓
  Progress Updates ←←←←←←←←←←←←←←←←←←←←←← Task Status
```

---

## Installation & Setup

### 1. Install Prefect
```bash
pip install prefect
```

### 2. Start Prefect Server
```bash
# Terminal 1 - Leave this running
prefect server start

# Server will start at:
# - UI: http://127.0.0.1:4200
# - API: http://127.0.0.1:4200/api
# - Docs: http://127.0.0.1:4200/docs
```

### 3. Configure Prefect API
```bash
# Terminal 2 - One-time setup
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Verify configuration
prefect config view
# Should show: PREFECT_API_URL='http://127.0.0.1:4200/api'
```

### 4. Verify Installation
```bash
# Check Prefect version
prefect version

# Test import
python -c "import prefect; print(f'Prefect {prefect.__version__} installed!')"
```

---

## Project Structure

### New Directory Structure
```
sculpt/
├── tasks/                          # NEW: Prefect tasks
│   ├── __init__.py
│   ├── umap_tasks.py              # ✅ COMPLETED
│   ├── genetic_tasks.py           # TODO
│   └── autoencoder_tasks.py       # TODO
├── flows/                          # NEW: Prefect flows
│   ├── __init__.py
│   ├── umap_flow.py               # ✅ COMPLETED
│   ├── genetic_flow.py            # TODO
│   └── autoencoder_flow.py        # TODO
├── callbacks/
│   ├── umap_callbacks.py          # ORIGINAL (keep for Graph 2/3)
│   ├── umap_prefect_callbacks.py  # ✅ NEW (Prefect-enabled)
│   ├── genetic_callbacks.py       # ORIGINAL
│   └── autoencoder_callbacks.py   # ORIGINAL
├── models/
│   └── deep_autoencoder.py        # Existing PyTorch model
├── utils/
│   ├── metrics/
│   │   ├── clustering_quality.py
│   │   ├── confidence_assessment.py
│   │   └── physics_features.py
│   └── ui.py
├── components/
│   └── layout.py                  # UI layout
└── app.py                          # Main Dash application

# Root level
test_prefect.py                     # ✅ Test script (working!)
docs/
└── PREFECT_MIGRATION_STATUS.md    # This file
```

---

## Implementation Progress

### ✅ Phase 1: Infrastructure Setup (COMPLETED)

**Status**: All infrastructure working perfectly

**What was done**:
1. Installed Prefect 3.4.23
2. Started Prefect server (Terminal 1)
3. Configured API endpoint
4. Created directory structure (`sculpt/tasks/`, `sculpt/flows/`)
5. Created `__init__.py` files
6. Verified with test flow

**Test Results**:
```bash
$ python test_prefect.py
✅ Test completed successfully!
   Embedding shape: (1000, 2)
   Computation time: 4.15s
📊 Check the Prefect UI at http://127.0.0.1:4200 to see the flow run
```

**Key Files Created**:
- `test_prefect.py` - Simple test to verify Prefect integration

---

### ✅ Phase 2: UMAP Tasks (COMPLETED)

**Status**: All UMAP tasks implemented and tested

**File**: `sculpt/tasks/umap_tasks.py`

**Tasks Created**:

1. **`load_and_prepare_data`**
   - Loads selected files from stored_files
   - Applies sampling (sample_frac parameter)
   - Adds file labels
   - Concatenates into combined dataframe
   - **Caching**: None (data loading is fast)
   - **Retries**: 1 (retry_delay: 10s)

2. **`compute_umap_embedding`**
   - Scales features using StandardScaler
   - Runs UMAP dimensionality reduction
   - Returns 2D embedding + computation time
   - **Caching**: 2 hours (based on data size + parameters)
   - **Retries**: 2 (retry_delay: 30s)
   - **Key Feature**: Most computationally expensive task

3. **`compute_clustering`**
   - Runs DBSCAN clustering on UMAP coordinates
   - Scales UMAP embedding before clustering
   - Returns cluster labels + clustering info
   - **Retries**: 1 (retry_delay: 10s)

4. **`create_umap_dataframe`**
   - Combines UMAP embedding, clusters, and original features
   - Creates final dataframe for visualization
   - **Retries**: 0 (simple data transformation)

**Key Implementation Details**:
- Uses `get_run_logger()` for Prefect-captured logging
- Type hints for better code clarity
- Comprehensive error handling
- Progress logging at each step

---

### ✅ Phase 3: UMAP Flow (COMPLETED)

**Status**: Flow orchestration implemented and tested

**File**: `sculpt/flows/umap_flow.py`

**Flows Created**:

1. **`umap_analysis_flow`** (Production flow)
   - Orchestrates all 4 UMAP tasks in sequence
   - Handles feature selection logic
   - Returns comprehensive results dictionary
   - **Retries**: 1 at flow level
   - **Parameters**:
     - `stored_files`: Dict of uploaded files
     - `selected_ids`: List of file IDs to process
     - `num_neighbors`: UMAP parameter
     - `min_dist`: UMAP parameter
     - `sample_frac`: Sampling fraction (1.0 = all data)
     - `selected_features_list`: List of feature groups
     - `dbscan_eps`: Clustering epsilon
     - `dbscan_min_samples`: Clustering min samples

2. **`test_umap_flow`** (Test flow)
   - Simple test with random data
   - Verifies Prefect integration
   - Used for debugging

**Flow Execution Steps**:
1. Load and prepare data
2. Compute UMAP embedding (longest step)
3. Compute DBSCAN clustering
4. Create final dataframe
5. Return comprehensive results

**Return Value Structure**:
```python
{
    'success': True,
    'combined_df': pd.DataFrame,      # Original data
    'umap_df': pd.DataFrame,          # UMAP results + clusters
    'cluster_labels': List[int],      # Cluster assignments
    'feature_cols': List[str],        # Features used
    'clustering_info': {
        'n_clusters': int,
        'n_noise': int,
        'eps': float,
        'min_samples': int
    },
    'debug_messages': List[str],
    'metadata': {
        'n_samples': int,
        'n_features': int,
        'n_clusters': int,
        'computation_time': float
    }
}
```

---

### 🔄 Phase 4: Dash Integration (IN PROGRESS)

**Status**: Callbacks created, testing integration with UI

**File**: `sculpt/callbacks/umap_prefect_callbacks.py`

**Callbacks Created**:

1. **`start_umap_flow`**
   - **Trigger**: "Run UMAP" button click
   - **Action**: Submits Prefect flow (non-blocking!)
   - **Returns**: 
     - Flow run ID (stored in dcc.Store)
     - Status message ("UMAP analysis started...")
   - **Key Feature**: UI remains responsive immediately after submission

2. **`monitor_umap_flow_progress`**
   - **Trigger**: Interval component (every 2 seconds)
   - **Action**: Checks Prefect flow status
   - **Returns**:
     - Progress bar value (0-100%)
     - Progress bar label
     - Progress bar color (info/primary/success/danger)
     - Detailed status (current task, task completion states)
     - UMAP visualization (when complete)
     - Debug output
     - Combined data store
     - Quality metrics
   - **States Handled**:
     - PENDING: "Flow queued, waiting to start..."
     - RUNNING: Shows current task + progress breakdown
     - COMPLETED: Displays UMAP graph + results
     - FAILED: Shows error message + link to Prefect UI

**UI Components Required** (need to be added to layout):
```python
# Progress tracking section
html.Div([
    html.H6("Analysis Progress"),
    html.Div(id="umap-status-message"),
    dbc.Progress(id="umap-progress-bar", value=0, striped=True, animated=True),
    html.Div(id="umap-status-details"),
])

# Update interval
dcc.Interval(id="umap-progress-interval", interval=2000, n_intervals=0)

# Flow run store
dcc.Store(id="umap-flow-run-store")
```

**Integration with app.py**:
```python
# Import new Prefect callbacks
from sculpt.callbacks.umap_prefect_callbacks import (
    start_umap_flow,
    monitor_umap_flow_progress,
)

# Comment out old callback to avoid conflicts
# from sculpt.callbacks.umap_callbacks import update_umap
```

**Next Steps for Dash Integration**:
1. [ ] Update layout file with progress UI components
2. [ ] Import callbacks in app.py
3. [ ] Comment out old update_umap callback
4. [ ] Test with real SCULPT data
5. [ ] Verify progress tracking works
6. [ ] Test UMAP visualization rendering

---

### ⏳ Phase 5: Genetic Programming Migration (TODO)

**Estimated Effort**: 4-6 hours

**Files to Create**:
- `sculpt/tasks/genetic_tasks.py`
- `sculpt/flows/genetic_flow.py`
- `sculpt/callbacks/genetic_prefect_callbacks.py`

**Current Implementation** (to migrate):
- File: `sculpt/callbacks/genetic_callbacks.py`
- Function: `run_genetic_feature_discovery_and_umap()`
- Complexity: HIGH (gplearn.genetic.SymbolicTransformer with 20+ generations)
- Typical Runtime: 30-300 seconds depending on generations/population

**Tasks to Create**:
1. `run_genetic_programming_task`
   - Setup: KMeans clustering for synthetic target
   - Core: SymbolicTransformer.fit_transform()
   - Extract symbolic expressions
   - **Caching**: 2 hours
   - **Retries**: 2

2. `compute_genetic_umap_task`
   - Run UMAP on genetic features
   - Similar to UMAP flow but different input

**Special Considerations**:
- Very long-running (can take 5+ minutes)
- Progress tracking per generation (if possible)
- Memory management (genetic programming is memory-intensive)
- Feature expression storage (symbolic formulas)

---

### ⏳ Phase 6: Autoencoder Migration (TODO)

**Estimated Effort**: 6-8 hours

**Files to Create**:
- `sculpt/tasks/autoencoder_tasks.py`
- `sculpt/flows/autoencoder_flow.py`
- `sculpt/callbacks/autoencoder_prefect_callbacks.py`

**Current Implementation** (to migrate):
- File: `sculpt/callbacks/autoencoder_callbacks.py`
- Function: `train_autoencoder_and_run_umap()`
- Complexity: VERY HIGH (PyTorch neural network training)
- Typical Runtime: 60-600 seconds depending on epochs

**Tasks to Create**:
1. `train_autoencoder_task`
   - PyTorch model initialization
   - Training loop with epoch progress
   - GPU/CPU device handling
   - Model state dict serialization
   - **Caching**: 1 hour (models are expensive)
   - **Retries**: 1 (training failures are often permanent)

2. `extract_latent_features_task`
   - Load trained model
   - Generate latent representations
   - Create latent feature dataframe

3. `compute_latent_umap_task`
   - Run UMAP on latent space

**Special Considerations**:
- **GPU Support**: Need to handle CUDA availability
- **Progress Tracking**: Per-epoch updates with loss values
- **Memory Management**: PyTorch memory leaks with long training
- **Model Persistence**: Save/load model state between tasks
- **Worker Configuration**: May need dedicated GPU worker queue

**PyTorch/Prefect Integration Notes**:
- Prefect tasks can run PyTorch code directly
- Device selection: `torch.device('cuda' if torch.cuda.is_available() else 'cpu')`
- Model state must be CPU-converted for serialization: `model.state_dict()`
- Consider using Prefect's artifact storage for large models

---

## Testing Strategy

### Test Flow (Working ✅)
```bash
python test_prefect.py
```

**What it tests**:
- Prefect server connectivity
- Task execution
- Flow orchestration
- Basic UMAP computation (1000 samples, 10 features)

**Expected Output**:
```
✅ Test completed successfully!
   Embedding shape: (1000, 2)
   Computation time: 4.15s
📊 Check the Prefect UI at http://127.0.0.1:4200 to see the flow run
```

### Integration Testing (Next)

**Test with Real SCULPT Data**:
1. Start Prefect server
2. Start Dash app
3. Upload actual COLTRIMS data files
4. Select files and features
5. Click "Run UMAP"
6. Verify:
   - Status message appears immediately
   - Progress bar updates every 2 seconds
   - UI remains responsive
   - Task progress shows correctly
   - UMAP graph appears when complete
   - Can navigate to other tabs while running

**Test Cases**:
- [ ] Single file, default features
- [ ] Multiple files, selected features
- [ ] Large dataset (>10,000 points)
- [ ] Different UMAP parameters
- [ ] Sampling (sample_frac < 1.0)
- [ ] Error handling (invalid data)
- [ ] Cancellation (close browser during run)

---

## Running the Application

### Development Mode

**Terminal 1** (Prefect Server):
```bash
prefect server start

# Leave this running
# UI available at: http://127.0.0.1:4200
```

**Terminal 2** (Dash App):
```bash
# Activate your virtual environment
source sculpt-test/bin/activate

# Start the application
python app.py

# App available at: http://127.0.0.1:9000
```

### Monitoring

**Prefect UI**: http://127.0.0.1:4200
- View all flow runs
- Click on runs to see detailed logs
- Monitor task execution timeline
- View error traces
- Check cache status

**SCULPT UI**: http://127.0.0.1:9000
- Normal Dash interface
- Progress bars show real-time status
- Links to Prefect UI for details

---

## Key Implementation Patterns

### Pattern 1: Task Definition
```python
from prefect import task, get_run_logger
from datetime import timedelta

@task(
    name="descriptive_task_name",
    description="What this task does",
    retries=2,
    retry_delay_seconds=30,
    cache_key_fn=lambda context, parameters, **kwargs: f"cache_{hash(str(parameters))}",
    cache_expiration=timedelta(hours=2)
)
def my_task(param1, param2):
    logger = get_run_logger()
    logger.info("Starting task...")
    
    # Do work
    result = expensive_computation(param1, param2)
    
    logger.info("Task complete!")
    return result
```

### Pattern 2: Flow Definition
```python
from prefect import flow, get_run_logger

@flow(
    name="My Analysis Flow",
    description="Complete analysis pipeline",
    retries=1
)
def my_analysis_flow(data, parameters):
    logger = get_run_logger()
    
    logger.info("Starting flow...")
    
    # Run tasks in sequence
    result1 = task1(data, parameters)
    result2 = task2(result1)
    result3 = task3(result2)
    
    logger.info("Flow complete!")
    
    return {
        'success': True,
        'results': result3,
        'metadata': {'runtime': 'etc'}
    }
```

### Pattern 3: Dash Callback (Flow Submission)
```python
@callback(
    Output("flow-store", "data"),
    Output("status", "children"),
    Input("run-button", "n_clicks"),
    State("parameters", "data"),
    prevent_initial_call=True
)
def start_flow(n_clicks, parameters):
    # Submit flow (non-blocking!)
    flow_run = my_analysis_flow.submit(
        data=parameters['data'],
        parameters=parameters
    )
    
    return (
        {"flow_run_id": str(flow_run.id)},
        "Analysis started..."
    )
```

### Pattern 4: Dash Callback (Progress Monitoring)
```python
@callback(
    Output("progress-bar", "value"),
    Output("results", "data"),
    Input("interval", "n_intervals"),
    State("flow-store", "data"),
    prevent_initial_call=True
)
def monitor_progress(n_intervals, flow_store):
    async def get_status():
        async with get_client() as client:
            flow_run = await client.read_flow_run(flow_store['flow_run_id'])
            return flow_run
    
    flow_run = asyncio.run(get_status())
    
    if flow_run.state.is_completed():
        result = flow_run.state.result()
        return 100, result
    
    return 50, {}  # Still running
```

---

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'prefect'"

**Cause**: Prefect not installed in current Python environment

**Solution**:
```bash
# Check which Python you're using
which python

# Install Prefect in that environment
python -m pip install prefect

# Or activate your virtual environment first
source venv_prefect/bin/activate
pip install prefect
```

### Issue 2: "Cannot connect to Prefect server"

**Cause**: Prefect server not running or API URL not configured

**Solution**:
```bash
# Terminal 1: Start server
prefect server start

# Terminal 2: Configure API
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api

# Verify
curl http://127.0.0.1:4200/api/health
```

### Issue 3: Flow doesn't appear in Prefect UI

**Cause**: Flow ran locally without server connection

**Solution**:
```bash
# Always ensure PREFECT_API_URL is set
prefect config view

# Should show: PREFECT_API_URL='http://127.0.0.1:4200/api'
```

### Issue 4: Dash callback doesn't trigger

**Cause**: Missing UI components or incorrect IDs

**Solution**:
- Verify all required components exist in layout
- Check component IDs match callback decorators exactly
- Look for JavaScript errors in browser console

### Issue 5: Progress bar doesn't update

**Cause**: Interval component disabled or flow_store empty

**Solution**:
```python
# Ensure Interval is enabled
dcc.Interval(
    id="progress-interval",
    interval=2000,
    disabled=False  # Must be False!
)

# Check flow_store has flow_run_id
print(flow_store)  # Should show: {'flow_run_id': '...'}
```

### Issue 6: "Task failed with unknown error"

**Cause**: Exception in task not properly caught

**Solution**:
- Check Prefect UI for full error trace
- Add try/except blocks in tasks
- Use `get_run_logger()` for debugging

---

## Performance Optimization

### Caching Strategy

**What to Cache**:
- ✅ UMAP embeddings (2 hours) - expensive computation
- ✅ Genetic programming results (2 hours) - very expensive
- ✅ Autoencoder training (1 hour) - extremely expensive
- ❌ Data loading - fast, not worth caching
- ❌ Simple transformations - overhead > benefit

**Cache Key Design**:
```python
# Good: Includes all parameters that affect output
cache_key_fn=lambda context, parameters, **kwargs: 
    f"umap_{len(parameters['data'])}_{parameters['n_neighbors']}_{parameters['min_dist']}_{hash(str(parameters['features']))}"

# Bad: Too general, different inputs get same cache
cache_key_fn=lambda context, parameters, **kwargs: "umap_cache"
```

### Task Parallelization (Future)

**Current**: All tasks run sequentially  
**Future**: Use Prefect's concurrent task runner

```python
from prefect.task_runners import ConcurrentTaskRunner

@flow(task_runner=ConcurrentTaskRunner())
def parallel_flow():
    # These could run in parallel
    result1 = task1.submit()
    result2 = task2.submit()
    
    # Wait for both
    return [result1.result(), result2.result()]
```

---

## Deployment Considerations

### For Academic Research Environment

**Current Setup** (Local Development):
- Prefect server on localhost
- Single worker (main process)
- Dash app on localhost:9000
- Good for: Development, single-user testing

**Recommended Production Setup**:
- Prefect Cloud (free tier) or self-hosted Prefect server
- Multiple workers (CPU + GPU if available)
- Dash app behind reverse proxy (nginx)
- Good for: Multi-user research lab, paper reproducibility

### Startup Script

**Create**: `start_sculpt.sh`
```bash
#!/bin/bash
set -e

echo "🚀 Starting SCULPT with Prefect..."

# Start Prefect server in background
prefect server start &
PREFECT_PID=$!

# Wait for Prefect to be ready
sleep 5

# Start Dash app
python app.py

# Cleanup on exit
trap "kill $PREFECT_PID" EXIT
```

**Usage**:
```bash
chmod +x start_sculpt.sh
./start_sculpt.sh
```

---

## Documentation Links

### Prefect Documentation
- Official Docs: https://docs.prefect.io/
- Task Guide: https://docs.prefect.io/concepts/tasks/
- Flow Guide: https://docs.prefect.io/concepts/flows/
- Caching: https://docs.prefect.io/concepts/tasks/#caching

### SCULPT Resources
- GitHub Repository: [Your repo URL]
- Paper: [Paper link if published]
- Prefect UI (local): http://127.0.0.1:4200

---

## Troubleshooting Checklist

Before asking for help, verify:

- [ ] Prefect server is running (Terminal 1)
- [ ] PREFECT_API_URL is configured correctly
- [ ] Virtual environment is activated
- [ ] All dependencies installed (`pip list | grep prefect`)
- [ ] Test flow works (`python test_prefect.py`)
- [ ] Prefect UI accessible (http://127.0.0.1:4200)
- [ ] Browser console shows no JavaScript errors
- [ ] All required UI components exist in layout
- [ ] Component IDs match callback decorators

---

## Next Chat Context Template

**For continuing in a new chat, use this template**:

```
I'm continuing the Prefect migration for SCULPT (COLTRIMS data analysis platform).

COMPLETED:
- ✅ Prefect infrastructure setup (server running, API configured)
- ✅ UMAP tasks implemented (sculpt/tasks/umap_tasks.py)
- ✅ UMAP flow created (sculpt/flows/umap_flow.py)
- ✅ Prefect callbacks for Dash (sculpt/callbacks/umap_prefect_callbacks.py)
- ✅ Test flow working (4.15s computation time)

CURRENT STATUS:
- Phase 4 - Integrating UMAP Prefect callbacks with Dash UI
- Need to: [specific task]

FILES CREATED:
- sculpt/tasks/umap_tasks.py (4 tasks: load_data, compute_umap, clustering, create_df)
- sculpt/flows/umap_flow.py (umap_analysis_flow + test_umap_flow)
- sculpt/callbacks/umap_prefect_callbacks.py (start_umap_flow, monitor_progress)
- test_prefect.py (working test script)
- docs/PREFECT_MIGRATION_STATUS.md (this documentation)

NEXT STEPS:
1. Complete Dash UI integration
2. Test with real SCULPT data
3. Migrate genetic programming
4. Migrate autoencoder training

QUESTION: [Your specific question here]

See full documentation in docs/PREFECT_MIGRATION_STATUS.md for detailed context.
```

---

## Change Log

| Date | Author | Change |
|------|--------|--------|
| 2025-10-10 | Initial | Created documentation, completed UMAP tasks/flows |

---

## Notes

- **Virtual environment**: `sculpt-test`
- Project root: `/Users/hazem/Documents/GitHub/CoInML/src/`
- Python path issues resolved by using correct venv
- Homebrew Python deprecation warning is harmless (safe to ignore)
- Pydantic compatibility warning is harmless (Prefect 3.x with Pydantic 2.x)

---

**Last Updated**: October 10, 2025  
**Status**: UMAP tasks and flows complete, Dash integration in progress  
**Next Milestone**: Complete UMAP Dash integration and test with real data