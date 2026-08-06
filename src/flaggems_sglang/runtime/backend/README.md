## Multiple backend adaptations
### Introduction
The `flaggems_sglang` operator library provides the ability to access multiple backends. If you are a chip vendor and wish to integrate `your flaggems_sglang code` into our official main branch, you simply need to follow these steps to complete the process.

#### step 1:
Create a folder named after your vendor in the `FlagGems-sglang/src/flaggems_sglang/runtime/backend` directory, following the pattern `_vendorname`. For example, you can refer to the structure of `FlagGems-sglang/src/flaggems_sglang/runtime/backend/_nvidia`.

#### step 2:
Create the necessary files, including but not limited to `__init__.py`, `heuristics_config_utils.py`, `tune_configs.yaml`, as well as a folder named `ops`. This is an example under `_nvidia`:
```
_nvidia/
├── __init__.py
├── heuristics_config_utils.py
├── tune_configs.yaml
└── ops
    ├── __init__.py
    ├── add.py
    └── gelu.py
```

##### step 2.1  `__init__.py`

You can copy `FlagGems-sglang/src/flaggems_sglang/runtime/backend/_nvidia/__init__.py` and the ***only change*** you need to make is to configure the `VendorDescriptor` class:
```python
from backend_utils import VendorDescriptor

vendor_info = VendorDescriptor(
    vendor_name="xxx",
    device_name="xxx",
    device_query_cmd="xxx",
)
```

###### Necessary fields:
- `vendor_name` is your vendor name, like `nvidia`
- `device_name` is your device name, like `cuda`
- `device_query_cmd` is a command that can only be successfully executed on your vendor's device, like `nvidia-smi`

###### Optional fields:
- `dispatch_key`: The operator registration field of `torch.library.Library` in PyTorch, like `PrivateUse1`
- `triton_extra_name`: Triton extra module name (e.g., `hip`, `xpu`, `cann`)
- `fp64_enabled` / `bf16_enabled` / `int64_enabled`: dtype capability flags (default `True`)
- `tle_enabled`: whether the vendor exposes a TLE runtime hook (default `False`)

##### step 2.2  `ops`
The `ops` directory is where `vendor-customized operator` implementations are stored. For instance, if you want to create a custom `add` operation, you should place the implementation in `ops/add.py`. Following that, you should configure `ops/__init__.py` accordingly.
```python
from .add import add
from .gelu import gelu

__all__ = ["add", "gelu"]
```
