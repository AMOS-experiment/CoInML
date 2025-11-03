#!/usr/bin/env python
"""
Script to register SCULPT flows with Prefect server.
Run this after Docker containers are up.
"""

import asyncio
import os
import sys
from prefect import flow, serve

# Set the API URL for Docker
os.environ["PREFECT_API_URL"] = "http://localhost:4200/api"

# Import your flows
sys.path.insert(0, '/app')  # For Docker container
try:
    from sculpt.flows.umap_flow import umap_analysis_flow
    print("✓ Successfully imported umap_analysis_flow")
except ImportError as e:
    print(f"✗ Could not import flows. Make sure you're in the right directory: {e}")
    # Try local import if not in Docker
    try:
        sys.path.insert(0, '.')
        from sculpt.flows.umap_flow import umap_analysis_flow
        print("✓ Successfully imported umap_analysis_flow (local)")
    except ImportError as e:
        print(f"✗ Failed to import flows: {e}")
        sys.exit(1)


def register_flows():
    """Register all SCULPT flows with Prefect using the new 3.x API."""
    
    print("\n🚀 Registering SCULPT flows with Prefect...")
    
    # Create deployment for UMAP flow using the new serve() method
    try:
        deployment = umap_analysis_flow.serve(
            name="umap-analysis-deployment",
            tags=["sculpt", "umap", "ml"],
            description="UMAP analysis flow for SCULPT platform",
            parameters={},  # Default parameters (will be overridden at runtime)
            # Note: work_pool_name is handled differently in Prefect 3.x
        )
        
        print(f"✓ UMAP flow registered successfully")
        
        # You can add more flows here as you migrate them
        # genetic_flow.serve(name="genetic-analysis-deployment", ...)
        # autoencoder_flow.serve(name="autoencoder-analysis-deployment", ...)
        
        print("\n✅ All flows registered successfully!")
        print("📊 View them at: http://localhost:4200/deployments")
        
        return deployment
        
    except Exception as e:
        print(f"✗ Error registering flows: {e}")
        return None


if __name__ == "__main__":
    # Run the registration
    deployment = register_flows()
    if deployment:
        print(f"\n🎯 Deployment created: {deployment}")
    else:
        print("\n❌ Failed to create deployment")
        sys.exit(1)