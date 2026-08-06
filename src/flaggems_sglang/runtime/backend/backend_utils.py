# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass


# Metadata template, Each vendor needs to specialize
# instances of this template
@dataclass
class VendorDescriptor:
    """
    A dataclass to describe the vendor-specific information for a hardware backend.
    """

    vendor_name: str
    device_name: str
    device_query_cmd: str
    dispatch_key: str = None
    triton_extra_name: str = None
    trademark: str = None
    fp64_enabled: bool = True
    bf16_enabled: bool = True
    int64_enabled: bool = True
    tle_enabled: bool = False


# Keep VendorInfoBase as alias for backward compatibility
VendorInfoBase = VendorDescriptor
