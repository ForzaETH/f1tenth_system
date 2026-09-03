"""ROS node exposing the physical command actually allowed for the car."""

import math
import time

from ackermann_msgs.msg import AckermannDriveStamped
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from .actuation_limiter import actuation_is_enabled
from .actuation_limiter import ActuationLimiter
from .actuation_limiter import ActuationLimits


class ActuationManager(Node):
    """Limit selected Ackermann commands and enforce the joystick deadman."""

    def __init__(self):
        super().__init__('actuation_manager')

        self.declare_parameter('input_topic', 'ackermann_cmd')
        self.declare_parameter('output_topic', 'ackermann_cmd_applied')
        self.declare_parameter('output_rate', 75.0)
        self.declare_parameter('speed_min', 0.0)
        self.declare_parameter('speed_max', 0.0)
        self.declare_parameter('max_acceleration', 2.5)
        self.declare_parameter('max_deceleration', 2.5)
        self.declare_parameter('steering_min', 0.0)
        self.declare_parameter('steering_max', 0.0)
        self.declare_parameter('max_steering_rate', 3.2)
        self.declare_parameter('command_timeout', 0.2)
        self.declare_parameter('joy_timeout', 0.2)
        self.declare_parameter('deadman_buttons', [4, 5])
        self.declare_parameter('frame_id', 'base_link')

        self.output_rate = self._positive_parameter('output_rate')
        self.command_timeout = self._positive_parameter('command_timeout')
        self.joy_timeout = self._positive_parameter('joy_timeout')
        self.deadman_buttons = [
            int(button)
            for button in self.get_parameter('deadman_buttons').value
        ]
        if not self.deadman_buttons or min(self.deadman_buttons) < 0:
            raise ValueError('deadman_buttons must contain non-negative indices')

        limits = ActuationLimits(
            speed_min=float(self.get_parameter('speed_min').value),
            speed_max=float(self.get_parameter('speed_max').value),
            max_acceleration=self._positive_parameter('max_acceleration'),
            max_deceleration=self._positive_parameter('max_deceleration'),
            steering_min=float(self.get_parameter('steering_min').value),
            steering_max=float(self.get_parameter('steering_max').value),
            max_steering_rate=self._positive_parameter(
                'max_steering_rate'),
        )
        self.limiter = ActuationLimiter(limits)

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.publisher = self.create_publisher(
            AckermannDriveStamped, output_topic, 1)
        self.create_subscription(
            AckermannDriveStamped, input_topic, self._command_callback, 1)
        self.create_subscription(Joy, '/joy', self._joy_callback, 1)

        self.requested_speed = 0.0
        self.requested_steering = 0.0
        self.last_command_time = None
        self.last_joy_time = None
        self.deadman_held = False
        self.last_tick_time = time.monotonic()
        self.last_enabled = None
        self.create_timer(1.0 / self.output_rate, self._publish_applied_command)

    def _positive_parameter(self, name):
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f'{name} must be finite and positive')
        return value

    def _command_callback(self, message):
        speed = float(message.drive.speed)
        steering = float(message.drive.steering_angle)
        if not math.isfinite(speed) or not math.isfinite(steering):
            self.get_logger().error('Ignoring non-finite Ackermann command')
            return
        self.requested_speed = speed
        self.requested_steering = steering
        self.last_command_time = time.monotonic()
        if message.header.frame_id:
            self.frame_id = message.header.frame_id

    def _joy_callback(self, message):
        self.last_joy_time = time.monotonic()
        self.deadman_held = any(
            button < len(message.buttons) and message.buttons[button] == 1
            for button in self.deadman_buttons
        )

    def _publish_applied_command(self):
        now = time.monotonic()
        dt = max(0.0, now - self.last_tick_time)
        self.last_tick_time = now

        command_fresh = (
            self.last_command_time is not None
            and now - self.last_command_time <= self.command_timeout
        )
        joy_fresh = (
            self.last_joy_time is not None
            and now - self.last_joy_time <= self.joy_timeout
        )
        enabled = actuation_is_enabled(
            now,
            self.last_command_time,
            self.last_joy_time,
            self.deadman_held,
            self.command_timeout,
            self.joy_timeout,
        )
        self._report_enabled_change(enabled, command_fresh, joy_fresh)

        state = self.limiter.step(
            self.requested_speed,
            self.requested_steering,
            dt,
            enabled=enabled,
        )
        message = AckermannDriveStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.drive.speed = state.speed
        message.drive.steering_angle = state.steering_angle
        self.publisher.publish(message)

    def _report_enabled_change(self, enabled, command_fresh, joy_fresh):
        if enabled == self.last_enabled:
            return
        self.last_enabled = enabled
        if enabled:
            self.get_logger().info('Actuation enabled')
            return

        reasons = []
        if not command_fresh:
            reasons.append('no fresh command')
        if not joy_fresh:
            reasons.append('no fresh joystick message')
        elif not self.deadman_held:
            reasons.append('deadman released')
        self.get_logger().warning('Actuation disabled: ' + ', '.join(reasons))


def main(args=None):
    """Run the actuation manager node."""
    rclpy.init(args=args)
    node = ActuationManager()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
