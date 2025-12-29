from wcwidth import wcswidth


def print_aligned(info: dict):
    """
    对齐打印 info 中的键值对信息
    """
    # 计算所有标签的最大视觉宽度
    max_width = 0
    for label in info.keys():
        # wcswidth 计算字符串的终端显示宽度
        width = wcswidth(label)
        if width > max_width:
            max_width = width

    # 打印对齐的输出
    for label, value in info.items():
        current_width = wcswidth(label)
        # 计算需要填充的空格数
        padding_spaces = max_width - current_width
        # 使用 f-string 打印，标签后跟计算出的空格数，然后是冒号和值
        print(f"{label}{' ' * padding_spaces} : {value}")


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.2f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/1024**2:.2f} MB"
    else:
        return f"{size_bytes/1024**3:.2f} GB"