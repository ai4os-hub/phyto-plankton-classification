# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
"""
Module to set Jupyter access password
from the PASSWORD environment, if available
COPY of: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/jupyter_notebook_config.py
"""
import os
from IPython.lib import passwd

c = c  # pylint:disable=undefined-variable
c.NotebookApp.ip = (
    "0.0.0.0"  # https://github.com/jupyter/notebook/issues/3946
)
c.NotebookApp.port = int(os.getenv("PORT", 8888))
c.NotebookApp.open_browser = False

# sets a password if PASSWORD is set in the environment
if "PASSWORD" in os.environ:
    password = os.environ["PASSWORD"]
    if password:
        c.NotebookApp.password = passwd(password)
    else:
        c.NotebookApp.password = ""
        c.NotebookApp.token = ""
    del os.environ["PASSWORD"]
