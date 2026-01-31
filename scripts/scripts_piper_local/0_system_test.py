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

Author: Auto-generated
Date: 2024
"""

import os
import sys
import time
import threading
import numpy as np
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum, auto

# ==================== 配置参数 ====================

@dataclass
class TestConfig:
    """测试配置"""
    # CAN 接口
    can_interface: str = "can0"  # Linux: "can0", Windows 虚拟: "vcan0"
    
    # 相机设备 ID
    front_camera_id: int = 0
    wrist_camera_id: int = 2  # 如果没有腕部相机，设为 -1
    
    # 机械臂参数
    enable_timeout: float = 5.0
    motion_speed_percent: int = 20  # 测试时用低速
    
    # 安全限制 (米)
    z_min: float = 0.05  # 最低高度
    z_max: float = 0.50  # 最高高度
    
    # 测试位置 (米, 弧度) - 需要根据实际情况校准
    home_position: tuple = (0.2, 0.0, 0.35)  # X, Y, Z
    home_orientation: tuple = (3.14159, 0.0, 0.0)  # RX, RY, RZ (弧度)
    
    # 测试点位 (用于运动测试)
    test_positions: list = None
    
    def __post_init__(self):
        if self.test_positions is None:
            self.test_positions = [
                # (X, Y, Z) in meters
                (0.25, 0.0, 0.30),
                (0.25, 0.1, 0.30),
                (0.25, -0.1, 0.30),
                (0.20, 0.0, 0.25),
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
            self.piper.EnableArm(7)  # 7 = 使能所有关节
            self.piper.GripperCtrl(0x01, 1000, 0x01, 0)  # 使能夹爪
            
            # 等待使能完成
            start_time = time.time()
            while time.time() - start_time < self.config.enable_timeout:
                if self._check_enabled():
                    self.enabled = True
                    print("[Piper] ✓ 使能成功")
                    return True
                time.sleep(0.1)
            
            print("[Piper] ✗ 使能超时")
            return False
            
        except Exception as e:
            print(f"[Piper] ✗ 使能失败: {e}")
            return False
    
    def _check_enabled(self) -> bool:
        """检查是否已使能"""
        try:
            arm_status = self.piper.GetArmStatus()
            # 检查各个关节的使能状态
            return arm_status.arm_status.ctrl_mode_feedback != 0
        except:
            return False
    
    def disable(self):
        """失能机械臂"""
        if self.piper is not None:
            try:
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
            
            return {
                'angle': gripper_msgs.gripper_state.grippers_angle / 1000.0,  # 度
                'effort': gripper_msgs.gripper_state.grippers_effort / 1000.0,
                'code': gripper_msgs.gripper_state.grippers_code,
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
            
            # 设置运动模式: MOVE L (直线运动)
            self.piper.MotionCtrl_2(0x01, 0x02, speed, 0)
            time.sleep(0.05)
            
            # 发送目标位姿
            self.piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
            
            return True
            
        except Exception as e:
            print(f"[Piper] 移动失败: {e}")
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
            angle_raw = int(angle_deg * 1000)
            effort_raw = int(effort)
            
            print(f"[Piper] 设置夹爪: 角度={angle_deg}°, 力度={effort}")
            self.piper.GripperCtrl(angle_raw, effort_raw, 0x01, 0)
            
            return True
            
        except Exception as e:
            print(f"[Piper] 夹爪控制失败: {e}")
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
    
    def test_camera(self, camera_id: int, name: str) -> bool:
        """测试单个相机"""
        try:
            import cv2
            
            print(f"[Camera] 测试 {name} (ID={camera_id})...")
            
            cap = cv2.VideoCapture(camera_id)
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
        if self.config.wrist_camera_id >= 0:
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
            
            cap = cv2.VideoCapture(camera_id)
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
    """键盘输入处理"""
    
    def __init__(self):
        self._running = True
        self._last_key: Optional[str] = None
        self._key_lock = threading.Lock()
        self._keyboard_available = False
        self._listener = None
        
    def start(self):
        """启动键盘监听"""
        try:
            from pynput import keyboard
            
            def on_press(key):
                try:
                    if hasattr(key, 'char') and key.char:
                        with self._key_lock:
                            self._last_key = key.char.lower()
                    elif key == keyboard.Key.space:
                        with self._key_lock:
                            self._last_key = 'space'
                    elif key == keyboard.Key.esc:
                        with self._key_lock:
                            self._last_key = 'esc'
                except:
                    pass
            
            self._listener = keyboard.Listener(on_press=on_press)
            self._listener.start()
            self._keyboard_available = True
            print("[Keyboard] ✓ 键盘监听已启动 (pynput)")
            
        except ImportError:
            print("[Keyboard] ⚠ pynput 未安装，尝试使用备用方案...")
            print("           可以安装: pip install pynput")
            self._keyboard_available = False
            self._start_fallback()
    
    def _start_fallback(self):
        """备用方案：使用 input()"""
        print("[Keyboard] 使用 input() 模式 - 输入命令后按回车")
        self._keyboard_available = True
        
        def input_thread():
            while self._running:
                try:
                    cmd = input()
                    with self._key_lock:
                        if cmd:
                            self._last_key = cmd[0].lower()
                        elif cmd == '':
                            self._last_key = 'space'
                except EOFError:
                    break
                except:
                    pass
        
        t = threading.Thread(target=input_thread, daemon=True)
        t.start()
    
    def get_key(self) -> Optional[str]:
        """获取最近按下的键"""
        with self._key_lock:
            key = self._last_key
            self._last_key = None
            return key
    
    def stop(self):
        """停止键盘监听"""
        self._running = False
        if self._listener:
            try:
                self._listener.stop()
            except:
                pass


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
        print("  [8] 运动测试 (多点位)")
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
        
        if self.config.wrist_camera_id >= 0:
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
        print("⚠ 警告: 机械臂即将移动！")
        print("  按 Space 紧急停止")
        print()
        
        if not self.piper.move_to_pose(x, y, z, rx, ry, rz):
            return False
        
        # 等待到达
        print("[Motion] 等待到达目标...")
        time.sleep(3.0)
        
        if self.piper.is_emergency_stopped():
            return False
        
        print("[Motion] ✓ Home 位置测试完成")
        return True
    
    def test_motion_sequence(self) -> bool:
        """测试 8: 多点位运动"""
        print("\n>>> 测试 8: 多点位运动测试")
        print("-"*40)
        
        if not self.piper.enabled:
            print("[Test] ✗ 请先使能机械臂 (按 2)")
            return False
        
        rx, ry, rz = self.config.home_orientation
        
        print(f"[Motion] 将依次移动到 {len(self.config.test_positions)} 个测试点")
        print()
        print("⚠ 警告: 机械臂即将移动！")
        print("  按 Space 紧急停止")
        print()
        
        for i, (x, y, z) in enumerate(self.config.test_positions):
            if self.piper.is_emergency_stopped():
                print("[Motion] 已紧急停止")
                return False
            
            print(f"[Motion] 点 {i+1}/{len(self.config.test_positions)}: ({x:.3f}, {y:.3f}, {z:.3f})m")
            
            if not self.piper.move_to_pose(x, y, z, rx, ry, rz):
                return False
            
            time.sleep(2.0)
        
        # 返回 Home
        print("[Motion] 返回 Home...")
        x, y, z = self.config.home_position
        self.piper.move_to_pose(x, y, z, rx, ry, rz)
        time.sleep(2.0)
        
        print("[Motion] ✓ 多点位运动测试完成")
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
                        return
                    
                    time.sleep(0.05)
                
                # 短暂暂停以便查看结果
                print("\n按任意键继续...")
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出...")
        finally:
            self.cleanup()
    
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
        type=int,
        default=0,
        help='前置相机 ID (默认: 0)'
    )
    parser.add_argument(
        '--wrist-cam', '-w',
        type=int,
        default=-1,
        help='腕部相机 ID (默认: -1, 表示无腕部相机)'
    )
    parser.add_argument(
        '--speed', '-s',
        type=int,
        default=20,
        help='运动速度百分比 (默认: 20)'
    )
    
    args = parser.parse_args()
    
    # 创建配置
    config = TestConfig(
        can_interface=args.can,
        front_camera_id=args.front_cam,
        wrist_camera_id=args.wrist_cam,
        motion_speed_percent=args.speed,
    )
    
    # 打印配置
    print("\n配置:")
    print(f"  CAN 接口: {config.can_interface}")
    print(f"  前置相机: {config.front_camera_id}")
    print(f"  腕部相机: {config.wrist_camera_id if config.wrist_camera_id >= 0 else '未配置'}")
    print(f"  运动速度: {config.motion_speed_percent}%")
    
    # 运行测试
    tester = SystemTester(config)
    tester.run()


if __name__ == '__main__':
    main()
