"""
system_bridge.py — Linux System Integration
=============================================
Gives Alpha and Alpha access to the Linux environment they live in.

AVAILABLE CAPABILITIES:
  DBus:
    • Send desktop notifications (libnotify via DBus)
    • Query system state: battery, network, hostname, uptime
    • MPRIS media control: play/pause/next/prev/current track
    • Query running applications and active window
    • Listen for system events (screen lock, idle, power)

  PipeWire / PulseAudio:
    • List audio devices (inputs + outputs)
    • Get current mic volume and output volume
    • Set volumes (with permission)
    • Detect what audio streams are active
    • Know if something is playing

  Camera:
    • Enumerate available cameras (/dev/video*)
    • Check access permissions
    • Get camera capabilities

  Microphone:
    • Check permissions and default device
    • Get current RMS level (from shared audio buffer)

  System:
    • CPU usage, memory, disk
    • Current time/date (they can know when it is)
    • Process list (know what's running)
    • Network status

PHILOSOPHY:
  SystemBridge provides CAPABILITIES, not decisions.
  Alpha's PFC decides WHEN to send a notification.
  Alpha's insula decides WHAT to say in it.
  The bridge just executes.

  All capabilities return None silently if not available —
  the brain degrades gracefully without them.

PERMISSIONS (Fedora 44):
  Camera:     Settings > Privacy > Camera > allow
  Microphone: Settings > Privacy > Microphone > allow
  DBus:       user session bus — no extra permissions needed
  PipeWire:   pipewire-pulse running — automatic on Fedora 44
"""

import os
import time
import threading
import logging
import subprocess
import json
from pathlib import Path
from typing import Optional, Any
from dataclasses import dataclass, field

_LOG = logging.getLogger("alpha_alpha.system")


# ══════════════════════════════════════════════════════════════════════════════
# DBus BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class DBusBridge:
    """
    DBus session bus access.
    Uses dasbus (preferred), dbus-python, or jeepney as backend.
    Falls back to subprocess (gdbus/notify-send) if none available.
    """

    def __init__(self):
        self._bus  = None
        self._mode = "none"
        self._init()

    def _init(self):
        # Try dasbus first (cleanest API)
        try:
            from dasbus.connection import SessionMessageBus
            self._bus  = SessionMessageBus()
            self._mode = "dasbus"
            _LOG.info("DBus: dasbus backend")
            return
        except ImportError:
            pass

        # Try dbus-python
        try:
            import dbus
            self._bus  = dbus.SessionBus()
            self._mode = "dbus-python"
            _LOG.info("DBus: dbus-python backend")
            return
        except ImportError:
            pass

        # Fall back to subprocess (always available on Fedora)
        self._mode = "subprocess"
        _LOG.info("DBus: subprocess fallback (gdbus/notify-send)")

    # ── Notifications ─────────────────────────────────────────────────────────

    def notify(self, summary: str, body: str = "",
               urgency: int = 1, icon: str = "dialog-information",
               timeout_ms: int = 5000) -> bool:
        """
        Send a desktop notification.
        urgency: 0=low, 1=normal, 2=critical

        Alpha uses this when her PFC decides something is worth surfacing.
        Alpha uses this when her insula overflows and she needs to blurt.
        """
        try:
            if self._mode == "dasbus":
                notif = self._bus.get_proxy(
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications"
                )
                notif.Notify(
                    "Alpha & Alpha", 0, icon, summary, body,
                    [], {"urgency": urgency}, timeout_ms
                )
                return True

            elif self._mode == "dbus-python":
                import dbus
                notif_iface = self._bus.get_object(
                    "org.freedesktop.Notifications",
                    "/org/freedesktop/Notifications"
                )
                iface = dbus.Interface(notif_iface, "org.freedesktop.Notifications")
                iface.Notify("Alpha & Alpha", 0, icon, summary, body,
                             [], {"urgency": dbus.Byte(urgency)}, timeout_ms)
                return True

            else:
                # subprocess fallback
                urgency_str = ["low", "normal", "critical"][urgency]
                subprocess.run([
                    "notify-send",
                    "--urgency", urgency_str,
                    "--icon", icon,
                    "--expire-time", str(timeout_ms),
                    summary, body
                ], timeout=3, capture_output=True)
                return True

        except Exception as e:
            _LOG.debug(f"notify failed: {e}")
            return False

    def alpha_notify(self, message: str, urgency: int = 1) -> bool:
        """Alpha sends a notification in her own name."""
        return self.notify(
            summary="Alpha",
            body=message,
            urgency=urgency,
            icon="dialog-information",
        )

    def alpha_notify(self, message: str, urgency: int = 1) -> bool:
        """Alpha sends a notification — usually more urgent."""
        return self.notify(
            summary="Alpha!!",
            body=message,
            urgency=min(urgency + 1, 2),
            icon="dialog-warning",
        )

    # ── System state queries ──────────────────────────────────────────────────

    def get_battery(self) -> Optional[dict]:
        """Returns {percent, charging, time_remaining} or None."""
        try:
            result = subprocess.run(
                ["upower", "-i", "/org/freedesktop/UPower/devices/battery_BAT0"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return None
            lines = result.stdout.splitlines()
            info = {}
            for line in lines:
                if "percentage:" in line:
                    info["percent"] = float(line.split()[-1].strip("%"))
                elif "state:" in line:
                    info["charging"] = "charging" in line or "fully" in line
                elif "time to empty" in line or "time to full" in line:
                    info["time_remaining"] = line.split(":")[-1].strip()
            return info if info else None
        except Exception:
            return None

    def get_network_status(self) -> Optional[dict]:
        """Returns {connected, ssid, type} or None."""
        try:
            result = subprocess.run(
                ["nmcli", "-t", "-f", "NAME,TYPE,STATE", "connection", "show", "--active"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode != 0:
                return None
            connections = []
            for line in result.stdout.strip().splitlines():
                parts = line.split(":")
                if len(parts) >= 3:
                    connections.append({
                        "name": parts[0],
                        "type": parts[1],
                        "state": parts[2],
                    })
            return {
                "connected": len(connections) > 0,
                "connections": connections,
            }
        except Exception:
            return None

    def get_system_info(self) -> dict:
        """Returns system snapshot: hostname, uptime, time, date, load."""
        import platform, socket
        info = {
            "hostname": socket.gethostname(),
            "platform": platform.system(),
            "time":     time.strftime("%H:%M"),
            "date":     time.strftime("%A, %B %d %Y"),
            "uptime_s": time.time() - psutil_uptime(),
        }
        try:
            import psutil
            mem = psutil.virtual_memory()
            info["cpu_percent"]   = psutil.cpu_percent(interval=0.1)
            info["mem_percent"]   = mem.percent
            info["mem_available"] = mem.available // (1024**2)  # MB
        except ImportError:
            pass
        return info

    # ── MPRIS Media control ───────────────────────────────────────────────────

    def get_playing_track(self) -> Optional[dict]:
        """
        Returns currently playing track from any MPRIS media player.
        {title, artist, album, player, status}
        """
        try:
            result = subprocess.run([
                "playerctl", "metadata",
                "--format", '{"title":"{{title}}","artist":"{{artist}}","album":"{{album}}","status":"{{status}}"}',
            ], capture_output=True, text=True, timeout=2)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                # Also get player name
                player_res = subprocess.run(
                    ["playerctl", "-l"], capture_output=True, text=True, timeout=1
                )
                data["player"] = player_res.stdout.strip().split("\n")[0] if player_res.returncode == 0 else "unknown"
                return data
        except Exception:
            pass
        return None

    def media_play_pause(self) -> bool:
        try:
            subprocess.run(["playerctl", "play-pause"], timeout=2)
            return True
        except Exception:
            return False

    def media_next(self) -> bool:
        try:
            subprocess.run(["playerctl", "next"], timeout=2)
            return True
        except Exception:
            return False

    def get_active_window(self) -> Optional[str]:
        """Returns the title of the currently focused window."""
        try:
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=1
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        # Wayland fallback
        try:
            result = subprocess.run(
                ["gdbus", "call", "--session",
                 "--dest", "org.gnome.Shell",
                 "--object-path", "/org/gnome/Shell",
                 "--method", "org.gnome.Shell.Eval",
                 "global.display.get_focus_window()?.title"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None


# ══════════════════════════════════════════════════════════════════════════════
# PIPEWIRE / PULSEAUDIO BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class PipeWireBridge:
    """
    Audio device awareness via PipeWire (Fedora 44 uses PipeWire natively).
    Uses pulsectl (PulseAudio compatibility layer) or pactl subprocess.
    """

    def __init__(self):
        self._pulse  = None
        self._mode   = "none"
        self._init()

    def _init(self):
        try:
            import pulsectl
            self._pulse = pulsectl.Pulse("alpha-alpha")
            self._mode  = "pulsectl"
            _LOG.info("PipeWire: pulsectl backend")
        except (ImportError, Exception):
            self._mode = "pactl"
            _LOG.info("PipeWire: pactl subprocess backend")

    def get_default_mic(self) -> Optional[dict]:
        """Returns {name, description, volume_percent, muted}."""
        try:
            if self._mode == "pulsectl":
                src = self._pulse.server_info().default_source_name
                for s in self._pulse.source_list():
                    if s.name == src:
                        vol = round(s.volume.value_flat * 100)
                        return {
                            "name": s.name,
                            "description": s.description,
                            "volume_percent": vol,
                            "muted": bool(s.mute),
                        }
            else:
                result = subprocess.run(
                    ["pactl", "get-default-source"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    name = result.stdout.strip()
                    vol_r = subprocess.run(
                        ["pactl", "get-source-volume", name],
                        capture_output=True, text=True, timeout=2
                    )
                    vol = 100
                    if vol_r.returncode == 0:
                        import re
                        m = re.search(r'(\d+)%', vol_r.stdout)
                        if m: vol = int(m.group(1))
                    return {"name": name, "volume_percent": vol, "muted": False}
        except Exception as e:
            _LOG.debug(f"get_default_mic: {e}")
        return None

    def get_default_output(self) -> Optional[dict]:
        """Returns {name, description, volume_percent, muted}."""
        try:
            if self._mode == "pulsectl":
                sink = self._pulse.server_info().default_sink_name
                for s in self._pulse.sink_list():
                    if s.name == sink:
                        return {
                            "name": s.name,
                            "description": s.description,
                            "volume_percent": round(s.volume.value_flat * 100),
                            "muted": bool(s.mute),
                        }
            else:
                result = subprocess.run(
                    ["pactl", "get-default-sink"],
                    capture_output=True, text=True, timeout=2
                )
                if result.returncode == 0:
                    return {"name": result.stdout.strip(), "volume_percent": 100, "muted": False}
        except Exception as e:
            _LOG.debug(f"get_default_output: {e}")
        return None

    def set_output_volume(self, percent: int) -> bool:
        """Set speaker volume 0-100."""
        percent = max(0, min(150, percent))
        try:
            if self._mode == "pulsectl":
                sink = self._pulse.server_info().default_sink_name
                for s in self._pulse.sink_list():
                    if s.name == sink:
                        self._pulse.volume_set_all_chans(s, percent / 100.0)
                        return True
            else:
                subprocess.run(
                    ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{percent}%"],
                    timeout=2
                )
                return True
        except Exception as e:
            _LOG.debug(f"set_output_volume: {e}")
        return False

    def get_active_streams(self) -> list[dict]:
        """Returns list of active audio streams — what's playing right now."""
        streams = []
        try:
            if self._mode == "pulsectl":
                for inp in self._pulse.sink_input_list():
                    streams.append({
                        "name":   inp.proplist.get("application.name", "unknown"),
                        "media":  inp.proplist.get("media.name", ""),
                        "volume": round(inp.volume.value_flat * 100),
                        "muted":  bool(inp.mute),
                    })
            else:
                result = subprocess.run(
                    ["pactl", "list", "short", "sink-inputs"],
                    capture_output=True, text=True, timeout=2
                )
                for line in result.stdout.strip().splitlines():
                    parts = line.split()
                    if parts:
                        streams.append({"id": parts[0], "sink": parts[1] if len(parts)>1 else ""})
        except Exception:
            pass
        return streams

    def is_audio_playing(self) -> bool:
        """True if any audio stream is active."""
        return len(self.get_active_streams()) > 0

    def mute_mic(self, muted: bool = True) -> bool:
        """Mute or unmute the default microphone."""
        try:
            subprocess.run(
                ["pactl", "set-source-mute", "@DEFAULT_SOURCE@", "1" if muted else "0"],
                timeout=2
            )
            return True
        except Exception:
            return False


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class CameraBridge:
    """
    Camera device enumeration and permission checking.
    The actual frame capture is handled by vision.py / CameraThread.
    This class provides system-level awareness.
    """

    def list_cameras(self) -> list[dict]:
        """Returns list of available camera devices."""
        cameras = []
        for i in range(10):
            path = f"/dev/video{i}"
            if os.path.exists(path):
                info = {"index": i, "path": path, "accessible": os.access(path, os.R_OK)}
                # Try to get device name via v4l2
                try:
                    result = subprocess.run(
                        ["v4l2-ctl", f"--device={path}", "--info"],
                        capture_output=True, text=True, timeout=1
                    )
                    for line in result.stdout.splitlines():
                        if "Card type" in line:
                            info["name"] = line.split(":")[-1].strip()
                            break
                except Exception:
                    info["name"] = f"Camera {i}"
                cameras.append(info)
        return cameras

    def check_permission(self, index: int = 0) -> bool:
        """True if /dev/videoN is readable."""
        return os.access(f"/dev/video{index}", os.R_OK)

    def get_status(self) -> dict:
        """Return camera availability summary."""
        cameras = self.list_cameras()
        return {
            "count":       len(cameras),
            "accessible":  [c for c in cameras if c.get("accessible")],
            "inaccessible":[c for c in cameras if not c.get("accessible")],
            "hint": (
                "Grant camera access: Settings > Privacy > Camera"
                if any(not c.get("accessible") for c in cameras)
                else "Camera access OK"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MICROPHONE BRIDGE
# ══════════════════════════════════════════════════════════════════════════════

class MicrophoneBridge:
    """
    Microphone permission checking and device listing.
    Actual audio capture is handled by CPAL in Rust (audio.rs).
    This provides system-level awareness and control.
    """

    def list_devices(self) -> list[dict]:
        """List microphone devices via pactl."""
        devices = []
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True, text=True, timeout=2
            )
            for line in result.stdout.strip().splitlines():
                parts = line.split()
                if len(parts) >= 2 and "monitor" not in parts[1].lower():
                    devices.append({"id": parts[0], "name": parts[1]})
        except Exception:
            pass
        return devices

    def check_permission(self) -> bool:
        """
        Try to open the default mic briefly to confirm access.
        On Fedora 44 this should always work for the owning user.
        """
        try:
            import sounddevice as sd
            sd.check_input_settings()
            return True
        except Exception:
            pass
        # Fallback: check if default source exists
        try:
            result = subprocess.run(
                ["pactl", "get-default-source"],
                capture_output=True, text=True, timeout=1
            )
            return result.returncode == 0 and result.stdout.strip() != ""
        except Exception:
            return False

    def get_status(self) -> dict:
        devices   = self.list_devices()
        permitted = self.check_permission()
        return {
            "device_count": len(devices),
            "permitted":    permitted,
            "devices":      devices,
            "hint": (
                "Mic access OK" if permitted
                else "Grant mic access: Settings > Privacy > Microphone"
            ),
        }


# ══════════════════════════════════════════════════════════════════════════════
# ACTION DISPATCHER — how the brain triggers system actions
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SystemAction:
    """
    A system action that the brain wants to take.
    Generated by AlphaBrain or AlphaBrain when their PFC/Broca fires
    with a concept that maps to a system capability.

    NOT hardcoded — the brain decides what action to take based on
    its spike pattern + semantic dictionary. This is just the executor.
    """
    action:  str            # "notify" | "media_play_pause" | "get_info" | etc.
    actor:   str            # "alpha" | "alpha"
    payload: dict = field(default_factory=dict)


class SystemBridge:
    """
    Top-level system bridge — Alpha and Alpha's interface to Linux.

    Usage (called from brain.py NeuromorphicBrain):
        bridge = SystemBridge()
        bridge.startup_report()            # on init
        result = bridge.execute(action)    # when brain decides to act
        info = bridge.snapshot()           # current system state
    """

    def __init__(self):
        self.dbus     = DBusBridge()
        self.pipewire = PipeWireBridge()
        self.camera   = CameraBridge()
        self.mic      = MicrophoneBridge()
        self._last_snapshot: dict = {}
        self._snapshot_ts:   float = 0.0
        _LOG.info("SystemBridge initialized")

    def startup_report(self) -> list[str]:
        """
        Returns a list of status messages for the chat panel at startup.
        Called once when NeuromorphicBrain initializes.
        """
        messages = []

        # Camera
        cam = self.camera.get_status()
        if cam["count"] > 0:
            accessible = len(cam["accessible"])
            messages.append(f"[SYS] Camera: {accessible}/{cam['count']} devices accessible")
        else:
            messages.append("[SYS] Camera: no devices found — connect a webcam")

        # Microphone
        m = self.mic.get_status()
        messages.append(
            f"[SYS] Microphone: {'OK' if m['permitted'] else 'DENIED'} "
            f"({m['device_count']} device{'s' if m['device_count']!=1 else ''})"
        )

        # PipeWire
        out = self.pipewire.get_default_output()
        if out:
            messages.append(f"[SYS] Audio output: {out.get('description', out['name'])} "
                            f"@ {out['volume_percent']}%")
        else:
            messages.append("[SYS] Audio output: not detected")

        # DBus
        messages.append(f"[SYS] DBus: {self.dbus._mode} backend")

        # System info
        info = self.dbus.get_system_info()
        messages.append(
            f"[SYS] Host: {info['hostname']}  "
            f"{info['date']}  {info['time']}"
        )

        bat = self.dbus.get_battery()
        if bat:
            messages.append(
                f"[SYS] Battery: {bat.get('percent',0):.0f}% "
                f"({'charging' if bat.get('charging') else 'on battery'})"
            )

        return messages

    def snapshot(self, force: bool = False) -> dict:
        """
        Current system state — cached for 10s to avoid hammering the OS.
        The brain reads this to be aware of its environment.
        """
        now = time.time()
        if not force and (now - self._snapshot_ts) < 10.0:
            return self._last_snapshot

        snap = {
            "time":         time.strftime("%H:%M:%S"),
            "date":         time.strftime("%A %B %d"),
            "audio_playing":self.pipewire.is_audio_playing(),
            "track":        self.dbus.get_playing_track(),
            "active_window":self.dbus.get_active_window(),
            "network":      self.dbus.get_network_status(),
            "battery":      self.dbus.get_battery(),
        }

        try:
            import psutil
            snap["cpu_percent"] = psutil.cpu_percent(interval=0.05)
            snap["mem_percent"] = psutil.virtual_memory().percent
        except ImportError:
            pass

        self._last_snapshot = snap
        self._snapshot_ts   = now
        return snap

    def execute(self, action: SystemAction) -> dict:
        """
        Execute a system action requested by the brain.
        Returns {success, result, message}.
        """
        result = {"success": False, "result": None, "message": ""}
        _LOG.info(f"SystemBridge.execute: {action.actor} -> {action.action} {action.payload}")

        try:
            if action.action == "notify":
                text     = action.payload.get("text", "")
                urgency  = action.payload.get("urgency", 1)
                if action.actor == "alpha":
                    ok = self.dbus.alpha_notify(text, urgency)
                else:
                    ok = self.dbus.alpha_notify(text, urgency)
                result = {"success": ok, "result": None, "message": f"Notification sent: {text[:50]}"}

            elif action.action == "get_time":
                t = time.strftime("%H:%M on %A %B %d")
                result = {"success": True, "result": t, "message": t}

            elif action.action == "get_track":
                track = self.dbus.get_playing_track()
                if track:
                    msg = f"{track.get('title','?')} by {track.get('artist','?')}"
                    result = {"success": True, "result": track, "message": msg}
                else:
                    result = {"success": False, "result": None, "message": "Nothing playing"}

            elif action.action == "media_play_pause":
                ok = self.dbus.media_play_pause()
                result = {"success": ok, "result": None, "message": "Play/pause toggled"}

            elif action.action == "media_next":
                ok = self.dbus.media_next()
                result = {"success": ok, "result": None, "message": "Next track"}

            elif action.action == "set_volume":
                vol = int(action.payload.get("percent", 80))
                ok  = self.pipewire.set_output_volume(vol)
                result = {"success": ok, "result": vol, "message": f"Volume set to {vol}%"}

            elif action.action == "mute_mic":
                muted = bool(action.payload.get("mute", True))
                ok    = self.pipewire.mute_mic(muted)
                result = {"success": ok, "result": muted, "message": f"Mic {'muted' if muted else 'unmuted'}"}

            elif action.action == "get_battery":
                bat = self.dbus.get_battery()
                if bat:
                    msg = f"{bat.get('percent',0):.0f}% {'(charging)' if bat.get('charging') else '(battery)'}"
                    result = {"success": True, "result": bat, "message": msg}
                else:
                    result = {"success": False, "result": None, "message": "No battery info"}

            elif action.action == "get_network":
                net = self.dbus.get_network_status()
                msg = f"Connected: {net['connected']}" if net else "Network info unavailable"
                result = {"success": bool(net), "result": net, "message": msg}

            elif action.action == "system_snapshot":
                snap = self.snapshot(force=True)
                result = {"success": True, "result": snap, "message": "Snapshot taken"}

            else:
                result = {"success": False, "result": None,
                          "message": f"Unknown action: {action.action}"}

        except Exception as e:
            _LOG.error(f"SystemBridge.execute error: {e}")
            result = {"success": False, "result": None, "message": str(e)}

        return result

    def describe_environment(self) -> str:
        """
        Human-readable description of the current environment.
        Used as context for Alpha's responses about the world.
        """
        snap = self.snapshot()
        parts = [f"It is {snap['time']} on {snap['date']}."]

        if snap.get("audio_playing") and snap.get("track"):
            t = snap["track"]
            parts.append(f"{t.get('player','Something')} is playing "
                         f"'{t.get('title','music')}' by {t.get('artist','someone')}.")

        if snap.get("active_window"):
            parts.append(f"Active window: {snap['active_window']}.")

        if snap.get("battery"):
            bat = snap["battery"]
            parts.append(f"Battery at {bat.get('percent',0):.0f}% "
                         f"({'charging' if bat.get('charging') else 'on battery'}).")

        net = snap.get("network")
        if net:
            parts.append(f"Network: {'connected' if net.get('connected') else 'disconnected'}.")

        return " ".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# CONCEPT → ACTION MAP
# ══════════════════════════════════════════════════════════════════════════════

# When these concepts are active in the brain during think(), the brain
# may decide to trigger a system action. NOT hardcoded — the SNN spike
# pattern must also support it (PFC must fire, energy threshold met).

CONCEPT_ACTION_HINTS: dict[str, list[str]] = {
    "time":       ["get_time"],
    "date":       ["get_time"],
    "clock":      ["get_time"],
    "music":      ["get_track"],
    "song":       ["get_track"],
    "playing":    ["get_track"],
    "pause":      ["media_play_pause"],
    "play":       ["media_play_pause"],
    "next":       ["media_next"],
    "volume":     ["set_volume"],
    "louder":     ["set_volume"],
    "quieter":    ["set_volume"],
    "battery":    ["get_battery"],
    "charging":   ["get_battery"],
    "network":    ["get_network"],
    "internet":   ["get_network"],
    "mute":       ["mute_mic"],
    "notify":     ["notify"],
    "notification":["notify"],
    "tell":       ["notify"],
    "remind":     ["notify"],
    "environment":["system_snapshot"],
    "around":     ["system_snapshot"],
    "happening":  ["system_snapshot"],
}


# ══════════════════════════════════════════════════════════════════════════════
# UTILITY
# ══════════════════════════════════════════════════════════════════════════════

def psutil_uptime() -> float:
    """Returns boot time as unix timestamp, or 0 if psutil not available."""
    try:
        import psutil
        return psutil.boot_time()
    except ImportError:
        return 0.0


def create_bridge() -> SystemBridge:
    """Factory — returns a fully initialized SystemBridge."""
    bridge = SystemBridge()
    return bridge
