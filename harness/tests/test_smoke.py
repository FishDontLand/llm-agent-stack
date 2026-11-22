import agent_harness
import agent_harness.agent


def test_package_importable():
    assert agent_harness is not None
    assert agent_harness.agent is not None
