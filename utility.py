import platform
def is_raspberry_pi() -> bool:
    return platform.machine().startswith('arm')  # Raspberry Pi uses ARM architecture


def is_windows()->bool:
    return platform.system() == 'Windows'


def get_horizontal_line_coordinates(width, height, position=0.5):

    y_position = int(height * position)
    start_point = (0, y_position)
    end_point = (width - 1, y_position)
    return [start_point, end_point]
