import platform
def is_raspberry_pi() -> bool:
    return platform.machine().startswith('arm')  # Raspberry Pi uses ARM architecture


def is_windows()->bool:
    return platform.system() == 'Windows'