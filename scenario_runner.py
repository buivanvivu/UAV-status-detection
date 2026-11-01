#!/usr/bin/env python3
"""
Chay kich ban bay tu dong tren SITL/Gazebo va phat nhan (label) len topic /uav_label
de data_collector.py ghi cung telemetry.

Nhan gui di co dang:  "maneuver,fault,pass_id,seg_id"
  - maneuver : hover/ascend/descend/forward/backward/left/right/turn_left/turn_right
  - fault    : normal/motor_fault/wind/gps_glitch
  - "ignore" : doan chuyen tiep (on dinh, dang bat/tat loi, dang recenter) -> preprocess se bo

Chay (Terminal 4 dang chay data_collector.py):
    python3 scenario_runner.py --faults normal --repeat 5
    python3 scenario_runner.py --faults motor_fault wind gps_glitch --repeat 5

TRUOC KHI CHAY loi mo phong, kiem tra ten tham so SITL co ton tai:
    rosrun mavros mavparam dump /tmp/params.txt
    grep -E "SIM_(ENGINE|WIND|GPS)" /tmp/params.txt
Neu ten khac, sua trong FAULTS / NEUTRAL ben duoi. Script se DUNG neu tham so khong ton tai
(tranh gan nhan 'motor_fault' cho du lieu bay binh thuong).

GHI CHU SUA DOI (sau su co repeat=50 chay khong xong):
  1) RECENTER giua cac pass: forward/backward/left/right chi doi xung ve LY
     THUYET, sai so thuc te (do tre tang/giam toc, gio...) tich luy dan qua
     nhieu pass va co the cham nguong an toan hoac lam vehicle roi GUIDED.
     -> Sau moi pass, UAV duoc dua ve dung vi tri/do cao cat canh bang
     setpoint vi tri (/mavros/setpoint_position/local) truoc khi pass tiep
     theo bat dau. Day la co che best-effort, NEN KIEM TRA THU trong SITL
     truoc khi chay repeat lon, vi chuyen doi giua setpoint van toc va vi tri
     giua chung phu thuoc cach ArduPilot/MAVROS xu ly GUIDED tren may ban.
  2) RETRY cho set_mode/set_param: 1 lan service timeout thoang qua giua
     phien chay dai (nhieu pass) truoc day lam dung ca kich ban. Gio se thu
     lai toi da 3 lan truoc khi bao loi that.
  3) Guard do cao (z < 1m) truoc day dung ngay khi gap 1 mau nhieu duy nhat;
     gio yeu cau lech sustain vai tick lien tiep (~0.3s) moi coi la roi/cham
     dat that.
  4) Neu KHONG dat duoc khung BODY_NED cho setpoint van toc, truoc day chi
     canh bao roi chay tiep (huong forward/backward/left/right co the sai
     y nghia ma khong bao loi gi). Gio se DUNG kich ban ngay - vi du lieu
     gan nhan sai huong con te hon la mat 1 lan chay.
"""
import argparse
import copy
from datetime import datetime

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import ParamValue, State
from mavros_msgs.srv import (CommandBool, CommandTOL, ParamGet, ParamSet,
                             SetMavFrame, SetMode)
from std_msgs.msg import String

VEL, VZ, YAW_RATE = 1.5, 0.8, 0.5   # m/s, m/s, rad/s

# (vx, vy, vz, yaw_rate) theo khung BODY (FLU): x=truoc, y=trai, z=len
MANEUVERS = {
    "hover":      (0, 0, 0, 0),
    "ascend":     (0, 0, VZ, 0),
    "descend":    (0, 0, -VZ, 0),
    "forward":    (VEL, 0, 0, 0),
    "backward":   (-VEL, 0, 0, 0),
    "left":       (0, VEL, 0, 0),
    "right":      (0, -VEL, 0, 0),
    "turn_left":  (0, 0, 0, YAW_RATE),
    "turn_right": (0, 0, 0, -YAW_RATE),
}

# Tham so SITL dung de tao loi (KIEM TRA TEN theo phien ban ArduPilot cua ban)
FAULTS = {
    "normal": {},
    "motor_fault": {"SIM_ENGINE_FAIL": 1, "SIM_ENGINE_MUL": 0.6},
    "wind": {"SIM_WIND_SPD": 8, "SIM_WIND_DIR": 90, "SIM_WIND_TURB": 2},
    "gps_glitch": {"SIM_GPS_GLITCH_X": 0.00005},
}
# Gia tri trung tinh de tra ve sau moi doan
NEUTRAL = {
    "SIM_ENGINE_FAIL": 0, "SIM_ENGINE_MUL": 1.0,
    "SIM_WIND_SPD": 0, "SIM_WIND_DIR": 0, "SIM_WIND_TURB": 0,
    "SIM_GPS_GLITCH_X": 0,
}
IGNORE = "ignore,-,-,-"

ALT_ABORT_MARGIN = 1.0   # m
ALT_ABORT_TICKS = 6      # so tick lien tiep (~0.3s @ 20Hz) truoc khi coi la that


def build_plan(faults):
    """(maneuver, fault, thoi_luong_s, co_ghi_nhan)"""
    plan = []
    if "normal" in faults:
        plan += [(m, "normal", 12, True) for m in MANEUVERS]
    for f in faults:
        if f == "normal":
            continue
        plan += [("hover", f, 15, True),
                 ("forward", f, 12, True),
                 ("backward", "normal", 12, False)]  # bay ve, KHONG ghi nhan
    return plan


class Runner:
    def __init__(self, alt):
        self.alt = alt
        self.state = State()
        self.pose = None
        self.z = 0.0
        self.home_pose = None
        self.airborne = False
        self.tag = datetime.now().strftime("%H%M%S")
        self.pass_id = ""
        self.seg_n = 0
        self.seg_id = ""
        self._low_alt_ticks = 0

        rospy.Subscriber("/mavros/state", State, lambda m: setattr(self, "state", m))
        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb)
        self.pub_vel = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=1)
        self.pub_pos = rospy.Publisher("/mavros/setpoint_position/local", PoseStamped, queue_size=1)
        self.pub_lbl = rospy.Publisher("/uav_label", String, queue_size=1, latch=True)

        names = ["/mavros/set_mode", "/mavros/cmd/arming", "/mavros/cmd/takeoff",
                 "/mavros/param/get", "/mavros/param/set"]
        for n in names:
            rospy.wait_for_service(n)
        self.srv_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.srv_arm = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.srv_takeoff = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)
        self.srv_pget = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        self.srv_pset = rospy.ServiceProxy("/mavros/param/set", ParamSet)

    def _pose_cb(self, msg):
        self.pose = msg
        self.z = msg.pose.position.z

    # ---------- tien ich ----------
    def _retry(self, func, tries=3, delay=1.0, desc=""):
        """Goi func() toi da `tries` lan, cho `delay` s giua cac lan neu that
        bai. Chi raise sau khi da het luot retry."""
        last_exc = None
        for i in range(tries):
            try:
                return func()
            except Exception as e:
                last_exc = e
                rospy.logwarn(f"[runner] {desc} that bai lan {i + 1}/{tries}: {e}")
                if i < tries - 1:
                    rospy.sleep(delay)
        raise RuntimeError(f"{desc} that bai sau {tries} lan: {last_exc}")

    def set_mode(self, mode):
        def _do():
            if not self.srv_mode(0, mode).mode_sent:
                raise RuntimeError(f"khong chuyen duoc mode {mode}")
        self._retry(_do, desc=f"set_mode({mode})")

    def set_param(self, name, val):
        def _do():
            if not self.srv_pget(name).success:
                raise RuntimeError(f"tham so '{name}' khong ton tai - sua ten trong FAULTS/NEUTRAL")
            pv = ParamValue()
            if float(val).is_integer():
                pv.integer = int(val)
            else:
                pv.real = float(val)
            if not self.srv_pset(param_id=name, value=pv).success:
                raise RuntimeError(f"dat tham so {name}={val} that bai")
            rospy.loginfo(f"[runner] {name} = {val}")
        self._retry(_do, desc=f"set_param({name}={val})")

    def hold(self, secs, vel=(0, 0, 0, 0), label=IGNORE, guard=False):
        """Phat setpoint van toc 20 Hz + nhan trong `secs` giay."""
        rate = rospy.Rate(20)
        end = rospy.Time.now() + rospy.Duration(secs)
        while rospy.Time.now() < end and not rospy.is_shutdown():
            msg = TwistStamped()
            msg.header.stamp = rospy.Time.now()
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z = vel[:3]
            msg.twist.angular.z = vel[3]
            self.pub_vel.publish(msg)
            self.pub_lbl.publish(String(data=label))

            if self.airborne and self.z < ALT_ABORT_MARGIN:
                self._low_alt_ticks += 1
                if self._low_alt_ticks >= ALT_ABORT_TICKS:
                    raise RuntimeError(
                        f"Do cao < {ALT_ABORT_MARGIN} m trong {ALT_ABORT_TICKS} tick lien tiep: "
                        "UAV co the da roi/cham dat, dung kich ban")
            else:
                self._low_alt_ticks = 0

            if guard and ((self.z < 2.5 and vel[2] < 0) or self.z > 25):
                rospy.logwarn("[runner] cham gioi han do cao, ket thuc som doan nay")
                return
            rate.sleep()

    def recenter(self, timeout=15.0, tol=0.5):
        """Dua UAV ve dung vi tri/do cao cat canh bang setpoint vi tri, thay vi
        chi tin tuong forward/backward se tu trieu tieu sai so. Goi giua cac
        pass de sai so vi tri khong tich luy qua nhieu vong lap."""
        if self.home_pose is None:
            rospy.logwarn("[runner] chua co home_pose, bo qua recenter")
            return
        rospy.loginfo("[runner] recenter ve vi tri xuat phat...")
        target = copy.deepcopy(self.home_pose)
        rate = rospy.Rate(20)
        t0 = rospy.Time.now()
        while not rospy.is_shutdown():
            target.header.stamp = rospy.Time.now()
            self.pub_pos.publish(target)
            self.pub_lbl.publish(String(data=IGNORE))
            if self.pose is not None:
                dx = self.pose.pose.position.x - target.pose.position.x
                dy = self.pose.pose.position.y - target.pose.position.y
                dz = self.pose.pose.position.z - target.pose.position.z
                dist = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                if dist < tol:
                    rospy.loginfo(f"[runner] da ve gan vi tri xuat phat (lech {dist:.2f} m)")
                    break
            if (rospy.Time.now() - t0).to_sec() > timeout:
                rospy.logwarn(f"[runner] recenter qua {timeout}s, van con lech - "
                              "tiep tuc kich ban (neu lech tang dan qua nhieu pass, "
                              "kiem tra lai dong luc/gio hoac giam --repeat)")
                break
            rate.sleep()

    # ---------- quy trinh bay ----------
    def takeoff(self):
        while not self.state.connected and not rospy.is_shutdown():
            rospy.sleep(0.5)
        try:  # BODY_NED = 8: setpoint theo khung than may (x=truoc)
            rospy.wait_for_service("/mavros/setpoint_velocity/mav_frame", timeout=3)
            rospy.ServiceProxy("/mavros/setpoint_velocity/mav_frame", SetMavFrame)(8)
        except Exception as e:
            raise RuntimeError(
                f"Khong dat duoc mav_frame BODY_NED ({e}); huong "
                "forward/backward/left/right se khong con dung y nghia. Kiem tra "
                "mavros/apm.launch da chay dung chua roi thu lai."
            )
        self.set_mode("GUIDED")
        armed = False
        for _ in range(30):  # cho EKF/GPS san sang
            if self.srv_arm(True).success:
                armed = True
                break
            rospy.sleep(1.0)
        if not armed:
            raise RuntimeError("Khong arm duoc (EKF/GPS chua san sang?)")
        self.srv_takeoff(min_pitch=0, yaw=0, latitude=0, longitude=0, altitude=self.alt)
        t0 = rospy.Time.now()
        while self.z < 0.9 * self.alt and not rospy.is_shutdown():
            if (rospy.Time.now() - t0).to_sec() > 40:
                raise RuntimeError("Takeoff qua 40 s chua toi do cao")
            rospy.sleep(0.5)
        self.airborne = True
        rospy.sleep(1.0)  # cho pose on dinh truoc khi luu lam moc "home"
        self.home_pose = self.pose
        rospy.loginfo("[runner] da len do cao muc tieu, da luu vi tri home")

    def lab(self, man, fault):
        return f"{man},{fault},{self.pass_id},{self.seg_id}"

    def run_segment(self, man, fault, secs, record):
        self.seg_n += 1
        self.seg_id = f"{self.tag}s{self.seg_n}"
        params = FAULTS[fault]
        try:
            self.hold(2)                      # on dinh, khong nhan
            for k, v in params.items():
                self.set_param(k, v)
            if params:
                self.hold(3)                  # cho loi "ngam" vao, khong nhan
            label = self.lab(man, fault) if record else IGNORE
            rospy.loginfo(f"[runner] {self.seg_id}: {man} / {fault} / {secs}s "
                          f"{'(ghi nhan)' if record else '(bay ve, bo)'}")
            self.hold(secs, MANEUVERS[man], label, guard=True)
        finally:
            for k in params:
                try:
                    self.set_param(k, NEUTRAL[k])
                except Exception as e:
                    rospy.logerr(f"[runner] KHONG reset duoc {k}: {e}")
        if params:
            self.hold(4)                      # cho he thong hoi phuc, khong nhan

    def run(self, plan, repeat):
        self.takeoff()
        try:
            for p in range(1, repeat + 1):
                self.pass_id = f"{self.tag}p{p}"
                rospy.loginfo(f"[runner] ===== luot {p}/{repeat} =====")
                for man, fault, secs, rec in plan:
                    if rospy.is_shutdown():
                        return
                    self.run_segment(man, fault, secs, rec)
                if p < repeat:
                    self.recenter()
        finally:
            self.pub_lbl.publish(String(data=IGNORE))
            rospy.sleep(0.5)
            try:
                self.set_mode("LAND")
            except Exception as e:
                rospy.logerr(f"[runner] khong LAND duoc: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--faults", nargs="+", default=["normal"],
                    choices=list(FAULTS.keys()))
    ap.add_argument("--repeat", type=int, default=1, help="so luot lap lai ca kich ban")
    ap.add_argument("--alt", type=float, default=6.0, help="do cao takeoff (m)")
    args, _ = ap.parse_known_args()

    rospy.init_node("uav_scenario_runner", anonymous=True)
    Runner(args.alt).run(build_plan(args.faults), args.repeat)