# 定义颜色和样式的代码常量
class Colors:
    """ANSI 颜色代码"""
    RESET = '\033[0m'

    # 前景色
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'

    # 样式
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def colored_print(message, color_code, style_code=''):
    """
    打印彩色文本。
    """
    # 构造完整的转义序列：[样式;颜色m 文本 \033[0m
    print(f"{style_code}{color_code}{message}{Colors.RESET}")

def gov_print(message):
    """
    打印政府信息。
    """
    colored_print(message, Colors.BLUE, Colors.BOLD)

def dev_print(message):
    """
    打印开发信息。
    """
    colored_print(message, Colors.GREEN, Colors.BOLD)

def village_print(message):
    """
    打印村民信息。
    """
    colored_print(message, Colors.YELLOW, Colors.BOLD)