# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Simple setup script"""

import os
from setuptools import setup, find_packages

abspath = os.path.dirname(os.path.realpath(__file__))

with open("requirements.txt", encoding='utf-8') as f:
  requirements = f.read().splitlines()  # pylint: disable=invalid-name

print(find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]))

license_header = """#
# SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""

# Generate version file
with open(os.path.join(abspath, "version.txt"), encoding="utf-8") as f:
  version = f.read().strip()
with open(os.path.join(abspath, "distributed_embeddings/version.py"), "w", encoding="utf-8") as f:
  f.write(license_header)
  f.write(F"__version__ = \"{version}\"")

setup(
    name="distributed-embeddings",
    version=version,
    description="Distributed Embedding",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)
