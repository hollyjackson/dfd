import sys
from pathlib import Path
import pytest
import copy

# Add the parent directory to sys.path so we can import modules from dfd/
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import globals to manage state
import globals as global_state


@pytest.fixture(scope="function", autouse=True)
def reset_globals():
    """Reset global state before each test to prevent test interaction."""
    # Save current state
    saved_state = {}
    for attr in dir(global_state):
        if not attr.startswith('_') and attr not in ['init_NYUv2', 'init_MobileDepth', 'init_Make3D']:
            try:
                val = getattr(global_state, attr)
                if not callable(val):
                    saved_state[attr] = copy.deepcopy(val) if hasattr(val, '__deepcopy__') else val
            except:
                pass

    yield

    # Restore state after test
    for attr, val in saved_state.items():
        try:
            setattr(global_state, attr, val)
        except:
            pass
