"""Agents over latent context.

The haystack is split into 1,024-encoder-token segments, each compressed by the
LCLM and labelled with a segment id. The model sees every compressed segment in
one prompt and can call a native ``expand`` tool to read a segment's original
text. See ``agent/README.md``.
"""
