import ctypes
import psutil

# Constants
PROCESS_ALL_ACCESS = 0x1F0FFF

# Function to get the process ID
def get_process_id(process_name):
    for proc in psutil.process_iter(['pid', 'name']):
        if proc.info['name'] == process_name:
            return proc.info['pid']
    return None

# Function to read memory from a given process at a specific address
def read_memory(process_handle, address, data_type=ctypes.c_uint64):
    value = data_type()
    bytes_read = ctypes.c_size_t()
    result = ctypes.windll.kernel32.ReadProcessMemory(process_handle,ctypes.c_void_p(address),ctypes.byref(value),ctypes.sizeof(value),ctypes.byref(bytes_read))
    if result:
        return value.value
    else:
        error_code = ctypes.GetLastError()
        print(f"Failed to read memory. Error code: {error_code}")
        return None