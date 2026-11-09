"""Compatibility with the pinned OpenWAM prompt-cache implementation.

Its __getitem__ moves entries to the end. MutableMapping.popitem can call it
while iterating, so evict directly through OrderedDict instead. Remove this
adapter once the dependency provides the same fix.
"""

from collections import OrderedDict


def repair_prompt_cache(engine):
    def set_item(self, key, value):
        if key in self:
            self.move_to_end(key)
        OrderedDict.__setitem__(self, key, value)
        while len(self) > self._maxsize:
            OrderedDict.__delitem__(self, next(iter(self)))
            if not self._evict_warned:
                self._evict_warned = True
                engine.logger.warning("Prompt embedding cache reached its size limit")

    engine._BoundedPromptEmbedCache.__setitem__ = set_item
