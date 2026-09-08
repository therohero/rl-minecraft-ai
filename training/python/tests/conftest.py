"""Put `python/` on `sys.path` so the tests import the trainer modules
(`features`, `ppo_agent`, ...) the same way `train.py` does, without a
package install."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
