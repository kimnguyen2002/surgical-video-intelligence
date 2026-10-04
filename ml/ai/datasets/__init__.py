"""
On-disk corpus discovery.

The platform is built around surgical video that already exists on the machine
rather than around a download step. This package finds those corpora, describes
them uniformly, and hands the rest of the system a single vocabulary for
"a video that can be played, indexed, searched, and asked about".

Three corpora ship with the SurgVU release and are recognised automatically:

===============  ==========================================  ====================
Corpus           Annotation                                  Used for
===============  ==========================================  ====================
``surgvu24``     ``tools.csv`` / ``tasks.csv`` intervals      presence + step
``cat1``         COCO bounding boxes                         detection, mAP eval
``cat2``         question / answer pairs                     VQA
===============  ==========================================  ====================
"""

from .corpora import (
    Cat1Video,
    Cat2Case,
    CorpusRegistry,
    SurgVUCase,
    VideoAsset,
    registry,
)

__all__ = [
    "Cat1Video",
    "Cat2Case",
    "CorpusRegistry",
    "SurgVUCase",
    "VideoAsset",
    "registry",
]
