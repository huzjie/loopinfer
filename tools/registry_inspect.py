# -*- coding: utf-8 -*-
"""Tool: inspect the sampler / model registries."""

from loopinfer.core.sampler import list_samplers
from loopinfer.model.factory import list_models


def inspect():
    return {"samplers": list_samplers(), "models": list_models()}


if __name__ == "__main__":
    print(inspect())
