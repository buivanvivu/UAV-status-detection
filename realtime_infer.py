#!/usr/bin/env python3
"""
Du doan trang thai / hanh dong UAV theo thoi gian thuc tu model da train
(uav_state_model.joblib, scaler.joblib, label_encoder.joblib tao boi train_model.py).

Chay:
    python3 realtime_infer.py --model_dir ./model --window 20

QUAN TRONG: --window phai giong voi --window da dung khi chay preprocess.py,
neu khong so dac trung se khong khop voi model.
"""
import argparse
import collections

import joblib
import message_filters
import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import RCIn, RCOut
from std_msgs.msg import String

# Thu tu phai khop CHINH XAC voi FEATURE_COLS trong preprocess.py
FEATURE_ORDER = [
    "roll", "pitch", "yaw",
    "pos_n", "pos_e", "pos_d",
    "vel_n", "vel_e", "vel_d",
    "servo1", "servo2", "servo3", "servo4",
    "rc1", "rc2", "rc3", "rc4",
]
RATE_COLS_IDX = [0, 1, 5]  # roll, pitch, pos_d trong FEATURE_ORDER


class RealtimeUAVClassifier:
    def __init__(self, model_dir, window):
        self.clf = joblib.load(f"{model_dir}/uav_state_model.joblib")
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.le = joblib.load(f"{model_dir}/label_encoder.joblib")
        self.window = window
        self.buffer = collections.deque(maxlen=window)
        self.buffer_time = collections.deque(maxlen=window)

        self.pub = rospy.Publisher("/uav_state_prediction", String, queue_size=1)

        pose_sub = message_filters.Subscriber("/mavros/local_position/pose", PoseStamped)
        vel_sub = message_filters.Subscriber("/mavros/local_position/velocity_local", TwistStamped)
        rcin_sub = message_filters.Subscriber("/mavros/rc/in", RCIn)
        rcout_sub = message_filters.Subscriber("/mavros/rc/out", RCOut)
        ts = message_filters.ApproximateTimeSynchronizer(
            [pose_sub, vel_sub, rcin_sub, rcout_sub], queue_size=20, slop=0.15
        )
        ts.registerCallback(self._cb)

    def _cb(self, pose, vel, rcin, rcout):
        q = pose.pose.orientation
        roll, pitch, yaw = tft.euler_from_quaternion([q.x, q.y, q.z, q.w])
        pos_n, pos_e, pos_d = pose.pose.position.y, pose.pose.position.x, -pose.pose.position.z
        vel_n, vel_e, vel_d = vel.twist.linear.y, vel.twist.linear.x, -vel.twist.linear.z
        servo = list(rcout.channels[:4]) if len(rcout.channels) >= 4 else [0, 0, 0, 0]
        rc = list(rcin.channels[:4]) if len(rcin.channels) >= 4 else [0, 0, 0, 0]

        sample = [roll, pitch, yaw, pos_n, pos_e, pos_d, vel_n, vel_e, vel_d, *servo, *rc]
        self.buffer.append(sample)
        self.buffer_time.append(pose.header.stamp.to_sec())

        if len(self.buffer) == self.window:
            self._predict()

    def _predict(self):
        arr = np.array(self.buffer)
        t = np.array(self.buffer_time)

        feat = []
        for i in range(arr.shape[1]):
            col = arr[:, i]
            feat.extend([col.mean(), col.std(), col.min(), col.max()])
        dt = max(t[-1] - t[0], 1e-6)
        for idx in RATE_COLS_IDX:
            feat.append((arr[-1, idx] - arr[0, idx]) / dt)

        feat = np.array(feat).reshape(1, -1)
        feat_s = self.scaler.transform(feat)
        pred = self.clf.predict(feat_s)[0]
        label = self.le.inverse_transform([pred])[0]
        proba = self.clf.predict_proba(feat_s).max()

        rospy.loginfo(f"[uav_state] {label} (do tin cay {proba:.2f})")
        self.pub.publish(String(data=f"{label}:{proba:.2f}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="./model")
    ap.add_argument("--window", type=int, default=20)
    args, _ = ap.parse_known_args()

    rospy.init_node("uav_realtime_classifier", anonymous=True)
    RealtimeUAVClassifier(args.model_dir, args.window)
    rospy.spin()
