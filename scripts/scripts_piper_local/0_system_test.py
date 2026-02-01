#!/usr/bin/env python3
"""
Script 0: System Test & Calibration Tool
=========================================

首次运行前的系统测试脚本，用于验证：
1. CAN 接口连接
2. 机械臂通信
3. 相机捕获
4. 坐标系校准
5. 夹爪控制
6. 运动控制

键盘控制：
- 数字键 1-9: 触发对应测试
- Space: 紧急停止
- Q: 退出程序
- R: 重置/使能机械臂

"""

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum, auto

# ==================== 配置参数 ====================

@dataclass
class TestConfig:
    """测试配置"""
    # CAN 接口
    can_interface: str = "can0"  # Linux: "can0", Windows 虚拟: "vcan0"
    
    # 相机设备 ID
    front_camera_id: Union[int, str] = "Orbbec_Gemini_335L"
    wrist_camera_id: Union[int, str] = "Dabai_DC1"  # 如果没有腕部相机，设为 -1, if the writs camera is present, set to its ID(2).
    
    # 机械臂参数
    enable_timeout: float = 5.0
    motion_speed_percent: int = 20  # 测试时用低速
    
    # 安全限制 (米)
    z_min: float = 0.05  # 最低高度
    z_max: float = 0.50  # 最高高度

    # 运动完成判定
    motion_timeout_s: float = 10.0
    pose_tolerance_pos_m: float = 0.015  # 位置容差 (米) - 15mm
    pose_tolerance_rot_deg: float = 5.0  # 姿态容差 (度)

    # 运动模式 (0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L)
    move_mode: int = 0x00

    # 夹爪完成判定
    gripper_timeout_s: float = 3.0
    gripper_tolerance_deg: float = 50.0  # 夹住物体时无法完全闭合，放宽容差
    
    # 测试位置 (米, 弧度) - 需要根据实际情况校准
    # 注意: 姿态需要与机械臂当前构型兼容，否则会报 TARGET_POS_EXCEEDS_LIMIT
    home_position: tuple = (0.054, 0.0, 0.175)  # X, Y, Z
    home_orientation: tuple = (3.14, 1.2, 3.14)  # RX, RY, RZ (弧度) ~= (180°, 68.8°, 180°)
    desk_center_position: tuple = (0.25, 0.0, 0.16)  # X, Y, Z
    desk_center_orientation: tuple = (3.14, 0.3, 3.14)  # RX, RY, RZ (弧度) ~= (180°, 17.2°, 180°)
    
    # 抓取测试参数 (类似 1_collect_data_piper.py 的 FSM 逻辑)
    # 抓取位置 (pick) - 使用 desk_center_position，在 __post_init__ 中初始化
    pick_position: tuple = None
    # 放置位置 (place) - 相对于 pick 有一定偏移
    place_offset: tuple = (-0.15, -0.15, 0.0)  # 偏移量 (米) #最大随机范围是xy方向+—0.15m
    # 抓取高度 - 默认使用 desk_center_position 的 Z 值，在 __post_init__ 中初始化
    grasp_height: float = None
    # 悬停高度 (比抓取高度高)
    hover_height: float = 0.25  # 悬停时的 Z 高度 (米)
    # 抓取朝向 (让夹爪指向下方) - 使用 desk_center_orientation
    grasp_orientation: tuple = None
    
    # 测试点位 (用于运动测试) - 保留用于简单点位测试
    test_positions: list = None
    
    def __post_init__(self):
        # 抓取位置默认使用桌面中心
        if self.pick_position is None:
            self.pick_position = self.desk_center_position
        # 抓取高度默认使用桌面中心的 Z 值
        if self.grasp_height is None:
            self.grasp_height = self.desk_center_position[2]  # Z = 0.1757m
        # 抓取朝向默认使用桌面中心朝向
        if self.grasp_orientation is None:
            self.grasp_orientation = self.desk_center_orientation
        # 测试点位
        if self.test_positions is None:
            cx, cy, cz = self.desk_center_position
            self.test_positions = [
                # (X, Y, Z) in meters - around desk center
                (cx, cy, cz + 0.08),
                (cx + 0.05, cy, cz + 0.08),
                (cx - 0.05, cy, cz + 0.08),
                (cx, cy + 0.05, cz + 0.08),
                (cx, cy - 0.05, cz + 0.08),
                (cx, cy, cz + 0.03),
                (cx, cy, cz),
            ]


# ==================== 全局状态 ====================

class TestState(Enum):
    IDLE = auto()
    RUNNING = auto()
    EMERGENCY_STOP = auto()


# ==================== Piper 控制器 ====================

class PiperController:
    """Piper 机械臂控制器（测试版）"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.piper: Optional[Any] = None
        self.connected = False
        self.enabled = False
        self._emergency_stop = False
        
    def connect(self) -> bool:
        """连接机械臂"""
        try:
            from piper_sdk import C_PiperInterface_V2
            
            print(f"[Piper] 正在连接 CAN 接口: {self.config.can_interface}")
            self.piper = C_PiperInterface_V2(self.config.can_interface)
            self.piper.ConnectPort()
            
            # 等待连接稳定
            time.sleep(0.5)
            self.connected = True
            print("[Piper] ✓ 连接成功")
            return True
            
        except ImportError as e:
            print(f"[Piper] ✗ 无法导入 piper_sdk: {e}")
            print("       请确保已安装: pip install -e ../piper_sdk")
            return False
        except Exception as e:
            print(f"[Piper] ✗ 连接失败: {e}")
            return False
    
    def enable(self) -> bool:
        """使能机械臂"""
        if not self.connected or self.piper is None:
            print("[Piper] ✗ 未连接，无法使能")
            return False
            
        try:
            print("[Piper] 正在使能机械臂...")
            
            # 发送使能命令
            self.piper.EnableArm(7)

            # 退出示教/拖动模式，并恢复控制模式
            self._ensure_control_mode()

            # 设置运动模式（MOVE_L）
            self.piper.MotionCtrl_2(0x01, self.config.move_mode, self.config.motion_speed_percent, 0)

            # 使能夹爪
            self.piper.GripperCtrl(0x01, 1000, 0x01, 0)
            
            # 等待使能完成
            start_time = time.time()
            while time.time() - start_time < self.config.enable_timeout:
                if self._check_enabled():
                    self.enabled = True
                    print("[Piper] ✓ 使能成功")
                    return True
                time.sleep(0.1)
            
            print("[Piper] ✗ 使能超时")
            self._print_arm_status(prefix="[Piper][Debug][EnableTimeout]")
            self._print_enable_status(prefix="[Piper][Debug][EnableTimeout]")
            return False
            
        except Exception as e:
            print(f"[Piper] ✗ 使能失败: {e}")
            return False

    def _ensure_control_mode(self):
        """确保退出示教模式并进入控制模式"""
        try:
            status = self._read_arm_status()
            if not status:
                return

            ctrl_mode_val = status.get('ctrl_mode_val')
        except Exception as e:
            print(f"[Piper][Debug] 控制模式恢复失败: {e}")
    
    def _check_enabled(self) -> bool:
        """检查是否已使能"""
        try:
            status = self._read_arm_status()
            if status is None:
                return False

            ctrl_mode_val = status.get('ctrl_mode_val')
            arm_state_val = status.get('arm_state_val')

            if ctrl_mode_val is not None:
                print(
                    f"[Piper][Debug] ctrl_mode={status.get('ctrl_mode')}({ctrl_mode_val}) "
                    f"arm_status={status.get('arm_state')}({arm_state_val})"
                )

            # ctrl_mode != 0 表示进入控制模式（CAN/以太网等）
            ctrl_ok = ctrl_mode_val is not None and ctrl_mode_val != 0
            enable_list = self._get_enable_status()
            enable_ok = bool(enable_list) and all(enable_list)
            if not enable_ok:
                self._print_enable_status(prefix="[Piper][Debug][Enable]")
            return ctrl_ok and enable_ok
        except:
            return False

    def _get_enable_status(self) -> Optional[list]:
        """获取各关节使能状态"""
        try:
            return self.piper.GetArmEnableStatus()
        except Exception:
            return None

    def _print_enable_status(self, prefix: str = "[Piper][Debug]"):
        """打印各关节使能状态"""
        status = self._get_enable_status()
        if status is None:
            print(f"{prefix} 无法读取关节使能状态")
            return
        enable_str = ", ".join([f"J{i+1}={'ON' if v else 'OFF'}" for i, v in enumerate(status)])
        print(f"{prefix} EnableStatus: {enable_str}")

    def _read_arm_status(self) -> Optional[Dict[str, Any]]:
        """读取机械臂状态（调试用）"""
        try:
            arm_status = self.piper.GetArmStatus()
            status = arm_status.arm_status
            ctrl_mode = getattr(status, 'ctrl_mode', None)
            arm_state = getattr(status, 'arm_status', None)
            mode_feed = getattr(status, 'mode_feed', None)
            teach_status = getattr(status, 'teach_status', None)
            motion_status = getattr(status, 'motion_status', None)

            return {
                'ctrl_mode': ctrl_mode,
                'arm_state': arm_state,
                'mode_feed': mode_feed,
                'teach_status': teach_status,
                'motion_status': motion_status,
                'ctrl_mode_val': int(ctrl_mode) if ctrl_mode is not None else None,
                'arm_state_val': int(arm_state) if arm_state is not None else None,
                'mode_feed_val': int(mode_feed) if mode_feed is not None else None,
                'teach_status_val': int(teach_status) if teach_status is not None else None,
                'motion_status_val': int(motion_status) if motion_status is not None else None,
            }
        except Exception:
            return None

    def _print_arm_status(self, prefix: str = "[Piper][Debug]"):
        """打印完整机械臂状态（调试用）"""
        status = self._read_arm_status()
        if not status:
            print(f"{prefix} 无法读取机械臂状态")
            return
        print(
            f"{prefix} ctrl_mode={status.get('ctrl_mode')}({status.get('ctrl_mode_val')}) "
            f"arm_status={status.get('arm_state')}({status.get('arm_state_val')}) "
            f"mode_feed={status.get('mode_feed')}({status.get('mode_feed_val')}) "
            f"teach_status={status.get('teach_status')}({status.get('teach_status_val')}) "
            f"motion_status={status.get('motion_status')}({status.get('motion_status_val')})"
        )
    
    def disable(self):
        """失能机械臂"""
        if self.piper is not None:
            try:
                if hasattr(self.piper, "DisablePiper"):
                    self.piper.DisablePiper()
                else:
                    self.piper.DisableArm(7)
                self.enabled = False
                print("[Piper] 已失能")
            except Exception as e:
                print(f"[Piper] 失能异常: {e}")
    
    def emergency_stop(self):
        """紧急停止"""
        self._emergency_stop = True
        if self.piper is not None:
            try:
                # 停止运动
                self.piper.MotionCtrl_2(0x01, 0x00, 0, 0)
                # 失能
                self.piper.DisableArm(7)
                self.enabled = False
                print("[Piper] !!! 紧急停止已触发 !!!")
            except Exception as e:
                print(f"[Piper] 紧急停止异常: {e}")
    
    def reset_emergency_stop(self):
        """重置紧急停止状态"""
        self._emergency_stop = False
        print("[Piper] 紧急停止已重置")
    
    def is_emergency_stopped(self) -> bool:
        """是否处于紧急停止状态"""
        return self._emergency_stop
    
    def get_tcp_pose(self) -> Optional[Dict[str, float]]:
        """获取当前 TCP 位姿"""
        if not self.connected or self.piper is None:
            return None
            
        try:
            end_pose = self.piper.GetArmEndPoseMsgs()
            
            # SDK 单位: 0.001mm -> 米
            x = end_pose.end_pose.X_axis / 1_000_000.0
            y = end_pose.end_pose.Y_axis / 1_000_000.0
            z = end_pose.end_pose.Z_axis / 1_000_000.0
            
            # SDK 单位: 0.001° -> 弧度
            rx = np.deg2rad(end_pose.end_pose.RX_axis / 1000.0)
            ry = np.deg2rad(end_pose.end_pose.RY_axis / 1000.0)
            rz = np.deg2rad(end_pose.end_pose.RZ_axis / 1000.0)
            
            return {
                'x': x, 'y': y, 'z': z,
                'rx': rx, 'ry': ry, 'rz': rz,
                'x_mm': x * 1000, 'y_mm': y * 1000, 'z_mm': z * 1000,
                'rx_deg': np.rad2deg(rx), 'ry_deg': np.rad2deg(ry), 'rz_deg': np.rad2deg(rz),
            }
        except Exception as e:
            print(f"[Piper] 读取位姿失败: {e}")
            return None
    
    def get_joint_angles(self) -> Optional[Dict[str, float]]:
        """获取当前关节角度"""
        if not self.connected or self.piper is None:
            return None
            
        try:
            joint_msgs = self.piper.GetArmJointMsgs()
            
            # SDK 单位: 0.001° -> 度
            angles = {}
            for i in range(1, 7):
                raw_angle = getattr(joint_msgs.joint_state, f'joint_{i}')
                angles[f'joint_{i}'] = raw_angle / 1000.0
            
            return angles
        except Exception as e:
            print(f"[Piper] 读取关节角度失败: {e}")
            return None
    
    def get_gripper_status(self) -> Optional[Dict[str, Any]]:
        """获取夹爪状态"""
        if not self.connected or self.piper is None:
            return None
            
        try:
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_state = gripper_msgs.gripper_state
            status_code = getattr(gripper_state, 'grippers_code', None)
            if status_code is None:
                status_code = getattr(gripper_state, 'status_code', None)
            
            return {
                'angle': gripper_state.grippers_angle / 1000.0,  # 度
                'effort': gripper_state.grippers_effort / 1000.0,
                'code': status_code,
            }
        except Exception as e:
            print(f"[Piper] 读取夹爪状态失败: {e}")
            return None
    
    def move_to_pose(self, x: float, y: float, z: float, 
                     rx: float, ry: float, rz: float,
                     speed_percent: int = None) -> bool:
        """移动到指定位姿
        
        Args:
            x, y, z: 位置 (米)
            rx, ry, rz: 姿态 (弧度)
            speed_percent: 速度百分比 (1-100)
        """
        if not self.enabled or self.piper is None:
            print("[Piper] ✗ 未使能，无法移动")
            return False
            
        if self._emergency_stop:
            print("[Piper] ✗ 紧急停止状态，无法移动")
            return False
        
        # 安全检查
        if z < self.config.z_min:
            print(f"[Piper] ✗ Z={z:.3f}m 低于安全限制 {self.config.z_min}m")
            return False
        if z > self.config.z_max:
            print(f"[Piper] ✗ Z={z:.3f}m 高于安全限制 {self.config.z_max}m")
            return False
        
        speed = speed_percent if speed_percent else self.config.motion_speed_percent
        
        try:
            # 确保处于控制模式
            self._ensure_control_mode()

            enable_list = self._get_enable_status()
            if enable_list is not None and not all(enable_list):
                self._print_enable_status(prefix="[Piper][Debug][BeforeMove]")
                print("[Piper] ✗ 关节未全部使能，取消运动")
                return False

            # 运动前状态快照
            self._print_arm_status(prefix="[Piper][Debug][BeforeMove]")

            # 转换单位
            # 米 -> 0.001mm
            X = int(x * 1_000_000)
            Y = int(y * 1_000_000)
            Z = int(z * 1_000_000)
            
            # 弧度 -> 0.001°
            RX = int(np.rad2deg(rx) * 1000)
            RY = int(np.rad2deg(ry) * 1000)
            RZ = int(np.rad2deg(rz) * 1000)
            
            print(f"[Piper] 移动到: X={x:.3f}m, Y={y:.3f}m, Z={z:.3f}m")
            print(f"        姿态: RX={np.rad2deg(rx):.1f}°, RY={np.rad2deg(ry):.1f}°, RZ={np.rad2deg(rz):.1f}°")
            print(
                "[Piper][Debug] 原始指令值: "
                f"X={X} Y={Y} Z={Z} RX={RX} RY={RY} RZ={RZ} speed={speed}"
            )
            
            # 设置运动模式: MOVE_P/MOVE_J/MOVE_L
            print(f"[Piper][Debug] MotionCtrl_2(cmd=0x01, mode={self.config.move_mode:#04x}, speed, 0)")
            self.piper.MotionCtrl_2(0x01, self.config.move_mode, speed, 0)
            time.sleep(0.05)
            
            # 发送目标位姿
            print("[Piper][Debug] EndPoseCtrl(X, Y, Z, RX, RY, RZ)")
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)

            # 运动后状态快照
            self._print_arm_status(prefix="[Piper][Debug][AfterMoveCmd]")
            
            return True
            
        except Exception as e:
            print(f"[Piper] 移动失败: {e}")
            return False

    def wait_until_pose(self, target: Dict[str, float], timeout_s: float,
                        pos_tol_m: float, rot_tol_deg: float) -> bool:
        """等待末端到达目标位姿（基于反馈位姿）"""
        start = time.time()
        last_pose = None
        last_debug = 0.0
        while time.time() - start < timeout_s:
            if self._emergency_stop:
                return False
            pose = self.get_tcp_pose()
            if pose is None:
                time.sleep(0.1)
                continue

            last_pose = pose
            dx = abs(pose['x'] - target['x'])
            dy = abs(pose['y'] - target['y'])
            dz = abs(pose['z'] - target['z'])
            drx = abs(pose['rx_deg'] - target['rx_deg'])
            dry = abs(pose['ry_deg'] - target['ry_deg'])
            drz = abs(pose['rz_deg'] - target['rz_deg'])

            if (dx <= pos_tol_m and dy <= pos_tol_m and dz <= pos_tol_m and
                    drx <= rot_tol_deg and dry <= rot_tol_deg and drz <= rot_tol_deg):
                return True

            # 每 1 秒打印一次调试信息
            now = time.time()
            if now - last_debug >= 1.0:
                print(
                    "[Piper][Debug][Tracking] "
                    f"Δx={dx:.3f} Δy={dy:.3f} Δz={dz:.3f} "
                    f"Δrx={drx:.1f} Δry={dry:.1f} Δrz={drz:.1f}"
                )
                self._print_arm_status(prefix="[Piper][Debug][Tracking]")
                last_debug = now

            time.sleep(0.1)

        if last_pose:
            print(
                "[Piper][Debug] 运动超时，当前位姿: "
                f"X={last_pose['x']:.3f} Y={last_pose['y']:.3f} Z={last_pose['z']:.3f} "
                f"RX={last_pose['rx_deg']:.1f} RY={last_pose['ry_deg']:.1f} RZ={last_pose['rz_deg']:.1f}"
            )
        return False
    
    def set_gripper(self, angle_deg: float, effort: int = 500) -> bool:
        """设置夹爪开度
        
        Args:
            angle_deg: 夹爪角度 (度), 0=全闭, 70=全开
            effort: 力度 (0-1000)
        """
        if not self.enabled or self.piper is None:
            print("[Piper] ✗ 未使能，无法控制夹爪")
            return False
            
        if self._emergency_stop:
            print("[Piper] ✗ 紧急停止状态，无法控制夹爪")
            return False
        
        try:
            # 确保处于控制模式
            self._ensure_control_mode()

            angle_raw = int(angle_deg * 1000)
            effort_raw = int(effort)
            
            print(f"[Piper] 设置夹爪: 角度={angle_deg}°, 力度={effort}")
            self.piper.GripperCtrl(angle_raw, effort_raw, 0x01, 0)

            # 闭环等待到位
            reached = self.wait_until_gripper(
                target_angle_deg=angle_deg,
                timeout_s=self.config.gripper_timeout_s,
                tol_deg=self.config.gripper_tolerance_deg,
            )
            if not reached:
                print("[Piper][Debug] 夹爪未在超时内到达目标")
            return reached
            
        except Exception as e:
            print(f"[Piper] 夹爪控制失败: {e}")
            return False

    def wait_until_gripper(self, target_angle_deg: float, timeout_s: float,
                           tol_deg: float) -> bool:
        """等待夹爪到达目标角度"""
        start = time.time()
        last_status = None
        while time.time() - start < timeout_s:
            if self._emergency_stop:
                return False
            status = self.get_gripper_status()
            if status is None:
                time.sleep(0.05)
                continue
            last_status = status
            err = abs(status['angle'] - target_angle_deg)
            if err <= tol_deg:
                return True
            time.sleep(0.05)

        if last_status is not None:
            print(
                "[Piper][Debug] 夹爪超时，当前角度: "
                f"{last_status['angle']:.1f}° (目标 {target_angle_deg:.1f}°)"
            )
        return False
    
    def open_gripper(self) -> bool:
        """打开夹爪"""
        return self.set_gripper(70.0, 500)
    
    def close_gripper(self) -> bool:
        """关闭夹爪"""
        return self.set_gripper(0.0, 500)
    
    def disconnect(self):
        """断开连接"""
        if self.enabled:
            self.disable()
        self.connected = False
        self.piper = None
        print("[Piper] 已断开连接")


# ==================== 相机测试 ====================

class CameraTest:
    """相机测试类"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.front_cam = None
        self.wrist_cam = None
    
    def _resolve_camera_source(self, camera_id: Union[int, str]):
        """将相机标识解析为可用的 VideoCapture 源

        支持:
        - 整数索引 (如 8)
        - /dev/videoX 路径
        - /dev/v4l/by-id/ 下的设备名或其子串 (如 Orbbec_Gemini_335L, Dabai_DC1)
        """
        if isinstance(camera_id, int):
            return camera_id

        if isinstance(camera_id, str):
            cam_str = camera_id.strip()
            if cam_str.isdigit():
                return int(cam_str)

            if cam_str.startswith("/dev/video") or cam_str.startswith("/dev/v4l/"):
                return cam_str

            try:
                import os
                from pathlib import Path

                by_id_dir = Path("/dev/v4l/by-id")
                if by_id_dir.exists():
                    matches = []
                    for p in by_id_dir.iterdir():
                        if cam_str in p.name:
                            matches.append(p)

                    # 优先选择 video-index0
                    matches.sort(key=lambda p: (
                        0 if "video-index0" in p.name else 1,
                        0 if "video-index1" in p.name else 1,
                        p.name,
                    ))

                    if matches:
                        return str(matches[0].resolve())
            except Exception:
                pass

        return camera_id

    def _is_camera_enabled(self, camera_id: Union[int, str]) -> bool:
        """判断相机是否启用（支持 -1 或 "-1"）"""
        if isinstance(camera_id, int):
            return camera_id >= 0
        if isinstance(camera_id, str):
            return camera_id.strip() != "-1"
        return False

    def _open_camera(self, camera_id: Union[int, str]):
        """优先使用 V4L2 打开，必要时回退到 obsensor 后端"""
        import cv2

        source = self._resolve_camera_source(camera_id)

        # 首选 V4L2，避免 OpenCV 走 Orbbec obsensor 后端
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap

        # 尝试 Orbbec obsensor 后端（如 Dabai / Gemini）
        obsensor_backend = getattr(cv2, "CAP_OBSENSOR", None)
        if obsensor_backend is not None:
            cap = cv2.VideoCapture(source, obsensor_backend)
            if cap.isOpened():
                return cap

        # 回退 CAP_ANY
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap

        return cap

    def test_camera(self, camera_id: int, name: str) -> bool:
        """测试单个相机"""
        try:
            import cv2
            
            print(f"[Camera] 测试 {name} (ID={camera_id})...")
            
            cap = self._open_camera(camera_id)
            if not cap.isOpened():
                print(f"[Camera] ✗ 无法打开 {name}")
                return False
            
            # 尝试读取几帧
            success_count = 0
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None:
                    success_count += 1
                time.sleep(0.1)
            
            cap.release()
            
            if success_count >= 3:
                print(f"[Camera] ✓ {name} 正常 ({success_count}/5 帧)")
                return True
            else:
                print(f"[Camera] ✗ {name} 读取不稳定 ({success_count}/5 帧)")
                return False
                
        except ImportError:
            print("[Camera] ✗ 无法导入 cv2，请安装: pip install opencv-python")
            return False
        except Exception as e:
            print(f"[Camera] ✗ 测试异常: {e}")
            return False
    
    def test_all_cameras(self) -> Dict[str, bool]:
        """测试所有相机"""
        results = {}
        
        # 测试前置相机
        results['front'] = self.test_camera(
            self.config.front_camera_id, 
            "前置相机"
        )
        
        # 测试腕部相机 (如果配置了)
        if self._is_camera_enabled(self.config.wrist_camera_id):
            results['wrist'] = self.test_camera(
                self.config.wrist_camera_id,
                "腕部相机"
            )
        else:
            print("[Camera] 跳过腕部相机测试 (未配置)")
            results['wrist'] = None
        
        return results
    
    def show_camera_preview(self, camera_id: int, name: str, duration: float = 5.0):
        """显示相机预览"""
        try:
            import cv2
            
            print(f"[Camera] 显示 {name} 预览 ({duration}秒)...")
            print("         按 Q 提前退出预览")
            
            cap = self._open_camera(camera_id)
            if not cap.isOpened():
                print(f"[Camera] ✗ 无法打开 {name}")
                return
            
            window_name = f"{name} Preview"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            start_time = time.time()
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if ret:
                    # 显示帧信息
                    h, w = frame.shape[:2]
                    cv2.putText(frame, f"{w}x{h}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name, frame)
                
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
            
            cap.release()
            cv2.destroyWindow(window_name)
            print(f"[Camera] {name} 预览结束")
            
        except Exception as e:
            print(f"[Camera] 预览异常: {e}")


# ==================== 键盘处理 ====================

class KeyboardHandler:
    """键盘输入处理（仅在终端输入时响应）"""
    
    def __init__(self):
        self._running = True
        self._last_key: Optional[str] = None
        self._key_lock = threading.Lock()
        self._keyboard_available = False
        
    def start(self):
        """启动键盘监听（使用 input() 模式，仅响应终端输入）"""
        print("[Keyboard] 使用终端输入模式 - 输入命令后按回车")
        self._keyboard_available = True
        
        def input_thread():
            while self._running:
                try:
                    cmd = input()
                    with self._key_lock:
                        if cmd:
                            self._last_key = cmd[0].lower()
                        # 空输入忽略
                except EOFError:
                    break
                except:
                    pass
        
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()
    
    def get_key(self) -> Optional[str]:
        """获取最近输入的命令"""
        with self._key_lock:
            key = self._last_key
            self._last_key = None
            return key
    
    def stop(self):
        """停止键盘监听"""
        self._running = False


# ==================== 主测试程序 ====================

class SystemTester:
    """系统测试主程序"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.piper = PiperController(config)
        self.camera_test = CameraTest(config)
        self.keyboard = KeyboardHandler()
        self.state = TestState.IDLE
        
    def print_menu(self):
        """打印菜单"""
        print("\n" + "="*60)
        print("  Piper 系统测试工具 - 首次运行检查")
        print("="*60)
        print()
        print("  [1] 测试 CAN 接口连接")
        print("  [2] 连接并使能机械臂")
        print("  [3] 读取当前位姿 (TCP + 关节角度)")
        print("  [4] 测试相机")
        print("  [5] 显示相机预览")
        print("  [6] 测试夹爪 (开->闭->开)")
        print("  [7] 移动到 Home 位置")
        print("  [8] 抓取放置测试 (Pick & Place FSM)")
        print("  [9] 运行全部测试")
        print()
        print("  [R] 重置/重新使能")
        print("  [D] 失能机械臂")
        print("  [Space] 紧急停止")
        print("  [Q] 退出程序")
        print()
        print("-"*60)
        status = "已使能" if self.piper.enabled else ("已连接" if self.piper.connected else "未连接")
        estop = " [紧急停止]" if self.piper.is_emergency_stopped() else ""
        print(f"  状态: {status}{estop}")
        print("-"*60)
        print()
    
    def test_can_interface(self) -> bool:
        """测试 1: CAN 接口"""
        print("\n>>> 测试 1: CAN 接口连接")
        print("-"*40)
        
        # 检查接口是否存在
        import subprocess
        import platform
        
        if platform.system() == 'Linux':
            try:
                result = subprocess.run(
                    ['ip', 'link', 'show', self.config.can_interface],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    print(f"[CAN] ✓ 接口 {self.config.can_interface} 存在")
                    if 'UP' in result.stdout:
                        print(f"[CAN] ✓ 接口已启动")
                        return True
                    else:
                        print(f"[CAN] ✗ 接口未启动，请运行:")
                        print(f"       sudo ip link set {self.config.can_interface} up type can bitrate 1000000")
                        return False
                else:
                    print(f"[CAN] ✗ 接口 {self.config.can_interface} 不存在")
                    print(f"       请检查 CAN 适配器连接")
                    return False
            except FileNotFoundError:
                print("[CAN] ⚠ 无法运行 ip 命令，跳过接口检查")
                return True
        else:
            print(f"[CAN] ⚠ Windows 系统，跳过 CAN 接口检查")
            print(f"       将在连接时验证")
            return True
    
    def test_robot_connection(self) -> bool:
        """测试 2: 机械臂连接"""
        print("\n>>> 测试 2: 机械臂连接与使能")
        print("-"*40)
        
        if not self.piper.connected:
            if not self.piper.connect():
                return False
        
        if not self.piper.enabled:
            if not self.piper.enable():
                return False
        
        return True
    
    def test_read_pose(self) -> bool:
        """测试 3: 读取位姿"""
        print("\n>>> 测试 3: 读取当前位姿")
        print("-"*40)
        
        if not self.piper.connected:
            print("[Test] ✗ 请先连接机械臂 (按 2)")
            return False
        
        # 读取 TCP 位姿
        pose = self.piper.get_tcp_pose()
        if pose:
            print("[TCP 位姿]")
            print(f"  位置: X={pose['x_mm']:.1f}mm, Y={pose['y_mm']:.1f}mm, Z={pose['z_mm']:.1f}mm")
            print(f"  姿态: RX={pose['rx_deg']:.1f}°, RY={pose['ry_deg']:.1f}°, RZ={pose['rz_deg']:.1f}°")
        else:
            print("[TCP] ✗ 读取失败")
        
        # 读取关节角度
        joints = self.piper.get_joint_angles()
        if joints:
            print("[关节角度]")
            for name, angle in joints.items():
                print(f"  {name}: {angle:.2f}°")
        else:
            print("[Joints] ✗ 读取失败")
        
        # 读取夹爪状态
        gripper = self.piper.get_gripper_status()
        if gripper:
            print("[夹爪状态]")
            print(f"  角度: {gripper['angle']:.1f}°")
            print(f"  力度: {gripper['effort']:.1f}")
            print(f"  状态码: {gripper['code']}")
        else:
            print("[Gripper] ✗ 读取失败")
        
        return pose is not None
    
    def test_cameras(self) -> bool:
        """测试 4: 相机"""
        print("\n>>> 测试 4: 相机连接")
        print("-"*40)
        
        results = self.camera_test.test_all_cameras()
        
        all_ok = True
        for name, result in results.items():
            if result is False:
                all_ok = False
        
        return all_ok
    
    def test_camera_preview(self):
        """测试 5: 相机预览"""
        print("\n>>> 测试 5: 相机预览")
        print("-"*40)
        
        self.camera_test.show_camera_preview(
            self.config.front_camera_id,
            "前置相机",
            duration=10.0
        )
        
        if self.camera_test._is_camera_enabled(self.config.wrist_camera_id):
            self.camera_test.show_camera_preview(
                self.config.wrist_camera_id,
                "腕部相机", 
                duration=10.0
            )
    
    def test_gripper(self) -> bool:
        """测试 6: 夹爪"""
        print("\n>>> 测试 6: 夹爪控制")
        print("-"*40)
        
        if not self.piper.enabled:
            print("[Test] ✗ 请先使能机械臂 (按 2)")
            return False
        
        print("[Gripper] 打开夹爪...")
        if not self.piper.open_gripper():
            return False
        time.sleep(1.5)
        
        if self.piper.is_emergency_stopped():
            return False
        
        print("[Gripper] 关闭夹爪...")
        if not self.piper.close_gripper():
            return False
        time.sleep(1.5)
        
        if self.piper.is_emergency_stopped():
            return False
        
        print("[Gripper] 打开夹爪...")
        if not self.piper.open_gripper():
            return False
        time.sleep(1.0)
        
        print("[Gripper] ✓ 夹爪测试完成")
        return True
    
    def test_move_home(self) -> bool:
        """测试 7: 移动到 Home"""
        print("\n>>> 测试 7: 移动到 Home 位置")
        print("-"*40)
        
        if not self.piper.enabled:
            print("[Test] ✗ 请先使能机械臂 (按 2)")
            return False
        
        x, y, z = self.config.home_position
        rx, ry, rz = self.config.home_orientation
        
        print(f"[Motion] Home 位置: ({x:.3f}, {y:.3f}, {z:.3f})m")
        print(f"[Motion] Home 姿态: ({np.rad2deg(rx):.1f}, {np.rad2deg(ry):.1f}, {np.rad2deg(rz):.1f})°")
        print()
        # Debug: 当前实时位姿与目标位姿对比
        current_pose = self.piper.get_tcp_pose()
        if current_pose is not None:
            dx = x - current_pose['x']
            dy = y - current_pose['y']
            dz = z - current_pose['z']
            drx = np.rad2deg(rx) - current_pose['rx_deg']
            dry = np.rad2deg(ry) - current_pose['ry_deg']
            drz = np.rad2deg(rz) - current_pose['rz_deg']
            print(
                "[Motion][Debug] 目标Home: "
                f"X={x:.3f} Y={y:.3f} Z={z:.3f} "
                f"RX={np.rad2deg(rx):.1f} RY={np.rad2deg(ry):.1f} RZ={np.rad2deg(rz):.1f}"
            )
            print(
                "[Motion][Debug] 当前TCP: "
                f"X={current_pose['x']:.3f} Y={current_pose['y']:.3f} Z={current_pose['z']:.3f} "
                f"RX={current_pose['rx_deg']:.1f} RY={current_pose['ry_deg']:.1f} RZ={current_pose['rz_deg']:.1f}"
            )
            print(
                "[Motion][Debug] 差值(Target-Current): "
                f"ΔX={dx:.3f} ΔY={dy:.3f} ΔZ={dz:.3f} "
                f"ΔRX={drx:.1f} ΔRY={dry:.1f} ΔRZ={drz:.1f}"
            )
        else:
            print("[Motion][Debug] 当前TCP位姿读取失败")
        print()
        print("⚠ 警告: 机械臂即将移动！")
        print("  按 Space 紧急停止")
        print()
        
        if not self.piper.move_to_pose(x, y, z, rx, ry, rz):
            return False
        
        # 等待到达
        print("[Motion] 等待到达目标...")
        target = {
            'x': x, 'y': y, 'z': z,
            'rx_deg': np.rad2deg(rx),
            'ry_deg': np.rad2deg(ry),
            'rz_deg': np.rad2deg(rz),
        }
        reached = self.piper.wait_until_pose(
            target,
            timeout_s=self.config.motion_timeout_s,
            pos_tol_m=self.config.pose_tolerance_pos_m,
            rot_tol_deg=self.config.pose_tolerance_rot_deg,
        )
        if not reached:
            print("[Motion] ✗ 未在超时内到达目标")
            return False
        
        if self.piper.is_emergency_stopped():
            return False
        
        print("[Motion] ✓ Home 位置测试完成")
        return True
    
    def test_motion_sequence(self) -> bool:
        """测试 8: 抓取放置流程测试 (模拟 Pick & Place FSM)
        
        流程参考 1_collect_data_piper.py:
        HOVER_PICK → LOWER_GRASP → CLOSE_GRIP → LIFT_OBJECT 
        → HOVER_PLACE → LOWER_PLACE → OPEN_GRIP → LIFT_RETREAT → HOME
        """
        print("\n>>> 测试 8: 抓取放置流程测试")
        print("-"*40)
        
        if not self.piper.enabled:
            print("[Test] ✗ 请先使能机械臂 (按 2)")
            return False
        
        # 抓取参数
        pick_x, pick_y, _ = self.config.pick_position
        place_x = pick_x + self.config.place_offset[0]
        place_y = pick_y + self.config.place_offset[1]
        grasp_z = self.config.grasp_height
        hover_z = self.config.hover_height
        rx, ry, rz = self.config.grasp_orientation
        
        print(f"[FSM] 抓取位置: ({pick_x:.3f}, {pick_y:.3f})m")
        print(f"[FSM] 放置位置: ({place_x:.3f}, {place_y:.3f})m")
        print(f"[FSM] 悬停高度: {hover_z:.3f}m, 抓取高度: {grasp_z:.3f}m")
        print()
        print("⚠ 警告: 机械臂即将执行抓取放置流程！")
        print("  按 Space 紧急停止")
        print()
        
        # 辅助函数：移动到位姿并等待
        def move_and_wait(x, y, z, state_name):
            if self.piper.is_emergency_stopped():
                return False
            print(f"[FSM] {state_name}: ({x:.3f}, {y:.3f}, {z:.3f})m")
            if not self.piper.move_to_pose(x, y, z, rx, ry, rz):
                return False
            target = {
                'x': x, 'y': y, 'z': z,
                'rx_deg': np.rad2deg(rx),
                'ry_deg': np.rad2deg(ry),
                'rz_deg': np.rad2deg(rz),
            }
            reached = self.piper.wait_until_pose(
                target,
                timeout_s=self.config.motion_timeout_s,
                pos_tol_m=self.config.pose_tolerance_pos_m,
                rot_tol_deg=self.config.pose_tolerance_rot_deg,
            )
            if not reached:
                print(f"[FSM] ✗ {state_name} 未在超时内到达")
                return False
            time.sleep(0.3)  # 稳定时间
            return True
        
        # ========== 抓取阶段 ==========
        # 1. HOVER_PICK: 悬停在抓取位置上方
        if not move_and_wait(pick_x, pick_y, hover_z, "HOVER_PICK"):
            return False
        
        # 2. 确保夹爪打开
        print("[FSM] OPEN_GRIP: 打开夹爪...")
        if not self.piper.open_gripper():
            print("[FSM] ✗ 打开夹爪失败")
            return False
        time.sleep(0.5)
        
        # 3. LOWER_GRASP: 下降到抓取高度
        if not move_and_wait(pick_x, pick_y, grasp_z, "LOWER_GRASP"):
            return False
        
        # 4. CLOSE_GRIP: 关闭夹爪抓取
        print("[FSM] CLOSE_GRIP: 关闭夹爪...")
        if self.piper.is_emergency_stopped():
            return False
        if not self.piper.close_gripper():
            print("[FSM] ✗ 关闭夹爪失败")
            return False
        time.sleep(0.5)
        
        # 5. LIFT_OBJECT: 抬起物体
        if not move_and_wait(pick_x, pick_y, hover_z, "LIFT_OBJECT"):
            return False
        
        # ========== 放置阶段 ==========
        # 6. HOVER_PLACE: 悬停在放置位置上方
        if not move_and_wait(place_x, place_y, hover_z, "HOVER_PLACE"):
            return False
        
        # 7. LOWER_PLACE: 下降到放置高度
        if not move_and_wait(place_x, place_y, grasp_z, "LOWER_PLACE"):
            return False
        
        # 8. OPEN_GRIP: 打开夹爪释放物体
        print("[FSM] OPEN_GRIP: 打开夹爪释放...")
        if self.piper.is_emergency_stopped():
            return False
        if not self.piper.open_gripper():
            print("[FSM] ✗ 打开夹爪失败")
            return False
        time.sleep(0.5)
        
        # 9. LIFT_RETREAT: 抬起撤离
        if not move_and_wait(place_x, place_y, hover_z, "LIFT_RETREAT"):
            return False
        
        # ========== 返回 Home ==========
        print("[FSM] GO_TO_HOME: 返回 Home...")
        x, y, z = self.config.home_position
        home_rx, home_ry, home_rz = self.config.home_orientation
        if not self.piper.move_to_pose(x, y, z, home_rx, home_ry, home_rz):
            return False
        target = {
            'x': x, 'y': y, 'z': z,
            'rx_deg': np.rad2deg(home_rx),
            'ry_deg': np.rad2deg(home_ry),
            'rz_deg': np.rad2deg(home_rz),
        }
        self.piper.wait_until_pose(
            target,
            timeout_s=self.config.motion_timeout_s,
            pos_tol_m=self.config.pose_tolerance_pos_m,
            rot_tol_deg=self.config.pose_tolerance_rot_deg,
        )
        
        print("[FSM] ✓ 抓取放置流程测试完成")
        return True
    
    def run_all_tests(self) -> Dict[str, bool]:
        """测试 9: 运行全部测试"""
        print("\n>>> 测试 9: 运行全部测试")
        print("="*60)
        
        results = {}
        
        # 1. CAN 接口
        results['can'] = self.test_can_interface()
        if not results['can']:
            print("\n[All] ✗ CAN 接口测试失败，停止后续测试")
            return results
        
        # 2. 机械臂连接
        results['robot'] = self.test_robot_connection()
        if not results['robot']:
            print("\n[All] ✗ 机械臂连接失败，停止后续测试")
            return results
        
        # 3. 读取位姿
        results['pose'] = self.test_read_pose()
        
        # 4. 相机
        results['camera'] = self.test_cameras()
        
        # 5. 夹爪
        results['gripper'] = self.test_gripper()
        if self.piper.is_emergency_stopped():
            print("\n[All] 已紧急停止")
            return results
        
        # 6. 运动
        results['motion'] = self.test_move_home()
        
        # 打印总结
        print("\n" + "="*60)
        print("  测试总结")
        print("="*60)
        for name, result in results.items():
            status = "✓ 通过" if result else "✗ 失败"
            print(f"  {name:12s}: {status}")
        print("="*60)
        
        return results
    
    def handle_emergency_stop(self):
        """处理紧急停止"""
        print("\n" + "!"*60)
        print("  !!! 紧急停止 !!!")
        print("!"*60)
        self.piper.emergency_stop()
        self.state = TestState.EMERGENCY_STOP
    
    def handle_reset(self):
        """处理重置"""
        print("\n>>> 重置机械臂")
        print("-"*40)
        
        if self.piper.is_emergency_stopped():
            self.piper.reset_emergency_stop()
        
        if self.piper.connected:
            self.piper.enable()
        else:
            self.piper.connect()
            self.piper.enable()
        
        self.state = TestState.IDLE
    
    def run(self):
        """主循环"""
        print("\n" + "#"*60)
        print("  启动 Piper 系统测试工具")
        print("#"*60)
        
        # 启动键盘监听
        self.keyboard.start()
        
        try:
            while True:
                self.print_menu()
                
                print("等待输入... (如果使用 input 模式，请输入命令后按回车)")
                
                # 等待按键
                while True:
                    key = self.keyboard.get_key()
                    
                    if key == 'space':
                        self.handle_emergency_stop()
                        break
                    
                    elif key == 'q':
                        print("\n退出程序...")
                        self._return_home_before_exit()
                        self.piper.disable()
                        return
                    
                    elif key == 'r':
                        self.handle_reset()
                        break
                    
                    elif key == 'd':
                        self.piper.disable()
                        break
                    
                    elif key == '1':
                        self.test_can_interface()
                        break
                    
                    elif key == '2':
                        self.test_robot_connection()
                        break
                    
                    elif key == '3':
                        self.test_read_pose()
                        break
                    
                    elif key == '4':
                        self.test_cameras()
                        break
                    
                    elif key == '5':
                        self.test_camera_preview()
                        break
                    
                    elif key == '6':
                        self.test_gripper()
                        break
                    
                    elif key == '7':
                        self.test_move_home()
                        break
                    
                    elif key == '8':
                        self.test_motion_sequence()
                        break
                    
                    elif key == '9':
                        self.run_all_tests()
                        break
                    
                    elif key == 'esc':
                        print("\n退出程序...")
                        self._return_home_before_exit()
                        self.piper.disable()
                        return
                    
                    time.sleep(0.05)
                
                # 短暂暂停以便查看结果
                print("\n按任意键继续...")
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出...")
            self._return_home_before_exit()
        finally:
            self.cleanup()
    
    def _return_home_before_exit(self):
        """退出前返回 Home 位置"""
        if not self.piper.enabled or self.piper.is_emergency_stopped():
            print("[Exit] 机械臂未使能或已紧急停止，跳过返回 Home")
            return
        
        print("[Exit] 返回 Home 位置...")
        x, y, z = self.config.home_position
        rx, ry, rz = self.config.home_orientation
        
        if self.piper.move_to_pose(x, y, z, rx, ry, rz):
            target = {
                'x': x, 'y': y, 'z': z,
                'rx_deg': np.rad2deg(rx),
                'ry_deg': np.rad2deg(ry),
                'rz_deg': np.rad2deg(rz),
            }
            reached = self.piper.wait_until_pose(
                target,
                timeout_s=self.config.motion_timeout_s,
                pos_tol_m=self.config.pose_tolerance_pos_m,
                rot_tol_deg=self.config.pose_tolerance_rot_deg,
            )
            if reached:
                print("[Exit] ✓ 已返回 Home 位置")
            else:
                print("[Exit] ⚠ 返回 Home 位置超时")
        else:
            print("[Exit] ⚠ 无法移动到 Home 位置")
    
    def cleanup(self):
        """清理资源"""
        print("\n清理资源...")
        self.keyboard.stop()
        if self.piper.connected:
            self.piper.disconnect()
        print("完成")


# ==================== 入口点 ====================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Piper 系统测试工具 - 首次运行检查"
    )
    parser.add_argument(
        '--can', '-c',
        default='can0',
        help='CAN 接口名称 (默认: can0)'
    )
    parser.add_argument(
        '--front-cam', '-f',
        default='Orbbec_Gemini_335L',
        help='前置相机 ID/名称 (默认: Orbbec_Gemini_335L)'
    )
    parser.add_argument(
        '--wrist-cam', '-w',
        default='Dabai_DC1',
        help='腕部相机 ID/名称 (默认: Dabai_DC1; 使用 -1 表示无腕部相机)'
    )
    parser.add_argument(
        '--speed', '-s',
        type=int,
        default=20,
        help='运动速度百分比 (默认: 20)'
    )
    
    args = parser.parse_args()
    
    # 创建配置
    def _normalize_cam_value(value):
        if isinstance(value, str):
            v = value.strip()
            if v == "-1":
                return -1
            if v.isdigit():
                return int(v)
            return v
        return value

    config = TestConfig(
        can_interface=args.can,
        front_camera_id=_normalize_cam_value(args.front_cam),
        wrist_camera_id=_normalize_cam_value(args.wrist_cam),
        motion_speed_percent=args.speed,
    )
    
    # 打印配置
    print("\n配置:")
    print(f"  CAN 接口: {config.can_interface}")
    print(f"  前置相机: {config.front_camera_id}")
    def _format_cam_value(value):
        if isinstance(value, int):
            return value if value >= 0 else '未配置'
        return value

    print(f"  腕部相机: {_format_cam_value(config.wrist_camera_id)}")
    print(f"  运动速度: {config.motion_speed_percent}%")
    
    # 运行测试
    tester = SystemTester(config)
    tester.run()


if __name__ == '__main__':
    main()
