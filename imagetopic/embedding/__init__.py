

from typing import Any
from imagetopic.embedding.generic import GENEmbedding

from imagetopic.embedding.image_strategy import IMAGEStrategy
from functools import partial

VLMEmbedding = partial(GENEmbedding, strategy_cls=IMAGEStrategy)
