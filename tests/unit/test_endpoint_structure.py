"""Structure API models: one byte cap shared with the contracts routes, edge field names."""

from __future__ import annotations


def test_one_request_byte_cap_everywhere():
    import models.contract_analysis_models as contracts
    import models.structure_models as structure

    assert structure.MAX_REQUEST_BYTES == contracts.MAX_REQUEST_BYTES
    assert structure.MAX_FILE_BYTES == contracts.MAX_FILE_BYTES


def test_service_and_message_edges_use_from_and_to_names():
    from models.structure_models import MessageEdgeOut, ServiceEdgeOut

    for model in (ServiceEdgeOut, MessageEdgeOut):
        fields = set(model.model_fields)
        sided = {f for f in fields if f.startswith(("from_", "to_"))}
        # Every endpoint-side field carries its side; only edge-level facts do not.
        assert fields - sided <= {"kind", "method", "match", "resolved_by", "candidates", "reason"}
        assert {"from_repo", "from_file", "from_line", "from_symbol", "to_repo"} <= sided
    assert {"from_channel", "from_via", "to_channel", "to_via"} <= set(MessageEdgeOut.model_fields)
    assert {"from_target", "to_route"} <= set(ServiceEdgeOut.model_fields)
