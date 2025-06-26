"""Basic tests for SCULPT"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_import():
    """Test that the main app can be imported"""
    try:
        import app
        assert True
    except ImportError as e:
        assert False, f"Failed to import app: {e}"

if __name__ == '__main__':
    test_import()
    print("Basic import test passed!")
