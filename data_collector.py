#!/usr/bin/env python3
"""
Thu telemetry UAV tu MAVROS, gan nhan THEO THOI GIAN tu topic /uav_label
(do scenario_runner.py phat ra). Ban nay thay the data_collector.py cu.

Chay:
    python3 data_collector.py --out ./data                 # dung cung scenario_runner.py
    python3 data_collector.py --out ./data --label hover   # che do thu cong: 1 nhan cho ca file

Cot nhan trong CSV: maneuver, fault, pass_id, seg_id
  - maneuver == "ignore" la doan chuyen tiep -> preprocess bo di.
  - Truoc khi runner bat dau, moi dong deu duoc gan "ignore".
"""
import argparse
import csv
import os
from datetime import datetime

import message_filters
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import RCIn, RCOut, State
from std_msgs.msg import String


class UAVDataCollector:
    def __init__(self, out_dir, fixed_label=None):
        os.makedirs(out_dir, exist_ok=True)
        tag = fixed_label if fixed_label else "scenario"
        self.filepath = os.path.join(
            out_dir, f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        self.csv_file = open(self.filepath, "w", newline="")
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow([
            "time", "roll", "pitch", "yaw",
            "pos_n", "pos_e", "pos_d",
            "vel_n", "vel_e", "vel_d",
            "servo1", "servo2", "servo3", "servo4",
            "rc1", "rc2", "rc3", "rc4",
            "armed", "mode",
            "maneuver", "fault", "pass_id", "seg_id",
        ])
        self.n_rows = 0

        if fixed_label:
            self.cur = [fixed_label, "normal", "manual", "0"]
        else:
            self.cur = ["ignore", "-", "-", "-"]
            rospy.Subscriber("/uav_label", String, self._label_cb)

        self.state = State()
        rospy.Subscriber("/mavros/state", State, lambda m: setattr(self, "state", m))

        subs = [
            message_filters.Subscriber("/mavros/local_position/pose", PoseStamped),
            message_filters.Subscriber("/mavros/local_position/velocity_local", TwistStamped),
            message_filters.Subscriber("/mavros/rc/in", RCIn),
            message_filters.Subscriber("/mavros/rc/out", RCOut),
        ]
        ts = message_filters.ApproximateTimeSynchronizer(subs, queue_size=20, slop=0.15)
        ts.registerCallback(self._sync_cb)

        rospy.Timer(rospy.Duration(5), self._watchdog, oneshot=True)
        rospy.loginfo(f"[data_collector] ghi vao: {self.filepath}")
        rospy.on_shutdown(self._on_shutdown)

    def _label_cb(self, msg):
        parts = msg.data.split(",")
        parts += ["-"] * (4 - len(parts))
        self.cur = parts[:4]

    def _watchdog(self, _evt):
        if self.n_rows == 0:
            rospy.logwarn("[data_collector] 5 s chua co dong nao. Kiem tra: "
                          "rostopic hz /mavros/rc/in  /mavros/rc/out  "
                          "/mavros/local_position/velocity_local")

    def _sync_cb(self, pose, vel, rcin, rcout):
        q = pose.pose.orientation
        roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        servo = list(rcout.channels[:4]) if len(rcout.channels) >= 4 else [0, 0, 0, 0]
        rc = list(rcin.channels[:4]) if len(rcin.channels) >= 4 else [0, 0, 0, 0]

        self.writer.writerow([
            pose.header.stamp.to_sec(), roll, pitch, yaw,
            pose.pose.position.y, pose.pose.position.x, -pose.pose.position.z,   # ENU -> N,E,D
            vel.twist.linear.y, vel.twist.linear.x, -vel.twist.linear.z,
            *servo, *rc,
            self.state.armed, self.state.mode,
            *self.cur,
        ])
        self.n_rows += 1
        if self.n_rows % 50 == 0:
            self.csv_file.flush()
            rospy.loginfo(f"[data_collector] {self.n_rows} dong | nhan hien tai: {','.join(self.cur)}")

    def _on_shutdown(self):
        self.csv_file.close()
        rospy.loginfo(f"[data_collector] Xong, {self.n_rows} dong -> {self.filepath}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--label", default=None, help="Che do thu cong: 1 nhan cho ca file")
    args, _ = ap.parse_known_args()

    rospy.init_node("uav_data_collector", anonymous=True)
    UAVDataCollector(args.out, args.label)
    rospy.spin()
