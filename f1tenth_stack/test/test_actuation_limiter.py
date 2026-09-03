"""Tests for physical-unit actuation limiting."""

import math

import pytest

from f1tenth_stack.actuation_limiter import actuation_is_enabled
from f1tenth_stack.actuation_limiter import ActuationLimiter
from f1tenth_stack.actuation_limiter import ActuationLimits


@pytest.fixture
def limiter():
    """Create representative forward-driving limits."""
    return ActuationLimiter(ActuationLimits(
        speed_min=0.0,
        speed_max=8.0,
        max_acceleration=2.0,
        max_deceleration=4.0,
        steering_min=-0.3,
        steering_max=0.3,
        max_steering_rate=1.0,
    ))


def test_acceleration_and_steering_are_rate_limited(limiter):
    state = limiter.step(5.0, 0.2, 0.1)

    assert state.speed == pytest.approx(0.2)
    assert state.steering_angle == pytest.approx(0.1)


def test_deceleration_uses_its_own_limit(limiter):
    limiter.step(5.0, 0.0, 1.0)
    state = limiter.step(0.0, 0.0, 0.25)

    assert state.speed == pytest.approx(1.0)


def test_targets_are_bounded_before_rate_limiting(limiter):
    state = limiter.step(20.0, -2.0, 10.0)

    assert state.speed == pytest.approx(8.0)
    assert state.steering_angle == pytest.approx(-0.3)


def test_disabled_state_stops_and_centres_safely(limiter):
    limiter.step(2.0, 0.3, 1.0)
    state = limiter.step(2.0, 0.3, 0.1, enabled=False)

    assert state.speed == 0.0
    assert state.steering_angle == pytest.approx(0.2)


def test_zero_dt_does_not_advance_output(limiter):
    state = limiter.step(2.0, 0.3, 0.0)

    assert state.speed == 0.0
    assert state.steering_angle == 0.0


@pytest.mark.parametrize('value', [math.inf, -math.inf, math.nan])
def test_non_finite_commands_are_rejected(limiter, value):
    with pytest.raises(ValueError):
        limiter.step(value, 0.0, 0.1)


def test_invalid_limits_are_rejected():
    with pytest.raises(ValueError):
        ActuationLimits(
            speed_min=2.0,
            speed_max=1.0,
            max_acceleration=1.0,
            max_deceleration=1.0,
            steering_min=-0.3,
            steering_max=0.3,
            max_steering_rate=1.0,
        )


def test_fresh_command_joystick_and_deadman_enable_actuation():
    assert actuation_is_enabled(10.0, 9.9, 9.85, True, 0.2, 0.2)


@pytest.mark.parametrize(
    'last_command,last_joy,deadman',
    [
        (None, 9.9, True),
        (9.9, None, True),
        (9.9, 9.9, False),
        (9.7, 9.9, True),
        (9.9, 9.7, True),
    ],
)
def test_missing_stale_or_released_inputs_disable_actuation(
        last_command, last_joy, deadman):
    assert not actuation_is_enabled(
        10.0, last_command, last_joy, deadman, 0.2, 0.2)
