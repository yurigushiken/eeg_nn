from importlib import import_module

TASKS = {}

def _register(name: str):
    mod = import_module(f"tasks.{name}")
    TASKS[name] = mod.label_fn

# Register only the four tasks in scope
for _name in (
    "cardinality_1_3",
    "cardinality_4_6",
    "cardinality_1_6",
    "landing_digit_1_3_within_small_and_cardinality",
    "landing_digit_4_6_within_large_and_cardinality",
    # Newly added tasks
    "landing_on_2_3",
    "landing_digit_1_3_within_small",
):
    _register(_name)

def get(name: str):
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Registered: {list(TASKS.keys())}")
    return TASKS[name]


