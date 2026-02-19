"""
Power management utilities to prevent system sleep during long-running operations.
"""
import platform
import ctypes


def prevent_sleep():
    """
    Prevent the system from going to sleep during execution.
    Currently supports Windows only.
    """
    system = platform.system()
    
    if system == 'Windows':
        # ES_CONTINUOUS: Informs the system that the state being set should remain in effect until the next call
        # ES_SYSTEM_REQUIRED: Forces the system to be in the working state by resetting the system idle timer
        # ES_DISPLAY_REQUIRED: Forces the display to be on by resetting the display idle timer
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        print("🔒 Power management: Sleep prevention enabled")
    else:
        print(f"⚠️  Power management: Sleep prevention not implemented for {system}")


def allow_sleep():
    """
    Allow the system to sleep normally (restore default power state).
    Currently supports Windows only.
    """
    system = platform.system()
    
    if system == 'Windows':
        ES_CONTINUOUS = 0x80000000
        
        # Reset to normal power state
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
        print("🔓 Power management: Sleep prevention disabled")
    else:
        # Silent on non-Windows systems
        pass
