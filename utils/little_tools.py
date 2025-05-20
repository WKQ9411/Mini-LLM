from wcwidth import wcswidth
import os
import matplotlib.pyplot as plt
import yaml


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


def create_folder(base_path):
    """
    创建文件夹，如果文件夹已存在，则在文件夹名称后添加数字后缀，返回添加了后缀的文件夹路径
    """
    folder_name = base_path
    counter = 1
    # 如果文件夹存在，尝试添加尾号
    while os.path.exists(folder_name):
        folder_name = f"{base_path}_{counter}"
        counter += 1
    # 创建文件夹
    os.makedirs(folder_name)
    return folder_name


def plot_loss_curve(total_loss, save_path):
    """
    绘制训练损失曲线并保存
    """
    plt.plot(total_loss)
    plt.xlabel("Iterations")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.savefig(save_path)


def save_to_yaml(dict, save_path):
    """
    将字典保存为 YAML 文件
    """
    with open(save_path, 'w') as file:
        yaml.dump(dict, file)


def load_yaml(file_path):
    """
    从 YAML 文件加载配置
    """
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    info = {
        "姓名": "张三",
        "Age": 30,
        "城市": "北京",
        "Occupation": "工程师 Engineer",
        "爱好 (Hobby)": "编程"
    }
    print_aligned(info)