"""Cross-service data flow contract registry for w2v pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FlowEdge:
    """One data-flow edge between two pipeline stages."""

    producer_model: str
    consumer_model: str
    label: str = ""
    forwarded_fields: set[str] | None = None  # None = check all


@dataclass
class FlowContract:
    """One producer -> consumer service pair with its edges."""

    name: str
    producer_service: str  # directory name, e.g. "analyzer_service"
    consumer_service: str
    edges: list[FlowEdge] = field(default_factory=list)


W2V_FLOW_CONTRACTS: list[FlowContract] = [
    FlowContract(
        name="analyze -> narrative (pages)",
        producer_service="analyzer_service",
        consumer_service="narrative_service",
        edges=[
            FlowEdge(
                producer_model="PageAnalysisResponse",
                consumer_model="PageInput",
                label="analyze.pages[] -> narrative.AnalysisInput.pages[]",
            ),
            FlowEdge(
                producer_model="SiteAnalysisResponse",
                consumer_model="SiteInput",
                label="analyze.site -> narrative.AnalysisInput.site",
            ),
        ],
    ),
    FlowContract(
        name="analyze -> script_generator (page_analysis)",
        producer_service="analyzer_service",
        consumer_service="script_generator_service",
        edges=[
            FlowEdge(
                producer_model="PageAnalysisResponse",
                consumer_model="PageAnalysisInput",
                label="analyze.pages[] -> generate.page_analysis[]",
            ),
        ],
    ),
    FlowContract(
        name="narrative -> script_generator (shotlist parts)",
        producer_service="narrative_service",
        consumer_service="script_generator_service",
        edges=[
            FlowEdge(
                producer_model="ShotListPartResponse",
                consumer_model="ShotDetailInput",
                label="shotlist.parts[] -> generate.parts[]",
            ),
        ],
    ),
    FlowContract(
        name="narrative -> video_compiler (shotlist)",
        producer_service="narrative_service",
        consumer_service="video_compiler_service",
        edges=[
            FlowEdge(
                producer_model="ShotListPartResponse",
                consumer_model="ShotListPartInput",
                label="shotlist.parts[] -> compile.shot_list.parts[]",
            ),
        ],
    ),
]
