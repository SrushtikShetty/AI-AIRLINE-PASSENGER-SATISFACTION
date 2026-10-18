import pytest

from src.data_validation import encode_target, validate_dataset
from tests.helpers import sample_frame


def test_validates_observed_schema_and_rating_boundaries():
    result = validate_dataset(sample_frame())
    assert result["valid"]
    assert all(check["status"] in {"pass", "warning"} for check in result["checks"])


def test_rejects_unknown_target_labels():
    frame = sample_frame()
    frame.loc[0, "Satisfaction"] = "Unknown"
    assert not validate_dataset(frame)["valid"]
    with pytest.raises(ValueError, match="Unknown or missing target"):
        encode_target(frame["Satisfaction"])


def test_rejects_out_of_range_service_ratings():
    frame = sample_frame()
    frame.loc[0, "Online Boarding"] = 6
    assert not validate_dataset(frame)["valid"]


def test_respects_feature_specific_rating_minimum():
    frame = sample_frame()
    frame.loc[0, "Baggage Handling"] = 0
    assert not validate_dataset(frame)["valid"]


def test_encode_target_uses_explicit_binary_mapping():
    encoded = encode_target(sample_frame()["Satisfaction"])
    assert set(encoded.unique()) == {0, 1}