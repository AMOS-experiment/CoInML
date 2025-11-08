"""
Test script to verify Prefect is working with SCULPT
"""

from sculpt.flows.umap_flow import test_umap_flow

if __name__ == "__main__":
    print("🧪 Testing Prefect integration...")

    # Run the test flow
    result = test_umap_flow(num_neighbors=15, min_dist=0.1)

    print("\n✅ Test completed successfully!")
    print(f"   Embedding shape: {result['embedding_shape']}")
    print(f"   Computation time: {result['computation_time']:.2f}s")
    print("\n📊 Check the Prefect UI at http://127.0.0.1:4200 to see the flow run!")
