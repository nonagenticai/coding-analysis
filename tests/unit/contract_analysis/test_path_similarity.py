from processing.contract_analysis.enrichment.path_similarity import segment_overlap_score


def test_identical_paths():
    assert segment_overlap_score("/api/foo/bar", "/api/foo/bar") == 1.0


def test_completely_different():
    assert segment_overlap_score("/api/alpha/beta", "/xyz/gamma/delta") < 0.2


def test_missing_prefix():
    """Client: /api/pdf/generate vs Service: /api/export/pdf/generate"""
    score = segment_overlap_score("/api/pdf/generate", "/api/export/pdf/generate")
    assert score > 0.5


def test_path_restructured():
    """Client: /api/elevation/{rack_id} vs Service: /api/racks/{rack_id}/elevation/{view_type}"""
    score = segment_overlap_score(
        "/api/elevation/{rack_id}", "/api/racks/{rack_id}/elevation/{view_type}"
    )
    assert score > 0.3


def test_param_normalization():
    """{id} and {item_id} should be treated the same."""
    assert segment_overlap_score("/api/foo/{id}", "/api/foo/{item_id}") == 1.0


def test_empty_path():
    assert segment_overlap_score("", "/api/foo") == 0.0
    assert segment_overlap_score("/api/foo", "") == 0.0


def test_terminal_segment_bonus():
    """Paths sharing a terminal segment should score higher than those that don't."""
    score_shared = segment_overlap_score("/api/pdf/generate", "/api/export/pdf/generate")
    score_diff = segment_overlap_score("/api/pdf/generate", "/api/export/pdf/status")
    assert score_shared > score_diff


def test_suffix_match_detected():
    """Client path is a suffix of service path."""
    score = segment_overlap_score("/api/detect", "/api/port-detection/detect/image")
    assert score > 0.2
