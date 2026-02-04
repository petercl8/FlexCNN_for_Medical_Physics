import sys
import os

def enable_system_wakelock():
    """
    Prevent Windows system sleep during long-running operations.
    Keeps the system awake but allows monitor to sleep independently.
    Only works on Windows; silently does nothing on other platforms.
    """
    if sys.platform != 'win32':
        return
    
    try:
        import ctypes
        # SetThreadExecutionState constants
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        
        # Set thread execution state to keep system awake
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        print("[INFO] System wakelock enabled - system will not sleep during long-running operations")
    except Exception as e:
        print(f"[WARNING] Could not enable system wakelock: {e}")

def disable_system_wakelock():
    """
    Restore normal Windows power management.
    Call after long-running operations complete.
    Only works on Windows; silently does nothing on other platforms.
    """
    if sys.platform != 'win32':
        return
    
    try:
        import ctypes
        # SetThreadExecutionState constants
        ES_CONTINUOUS = 0x80000000
        
        # Restore normal execution state (system can sleep normally)
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("[INFO] System wakelock disabled - normal power management restored")
    except Exception as e:
        print(f"[WARNING] Could not disable system wakelock: {e}")
