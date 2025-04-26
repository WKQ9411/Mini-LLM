import pkgutil
import importlib
import inspect
import os
from typing import Type, Tuple

# 给 get_model_and_args 函数添加类型提示
from .basemodel import BaseModel, BaseModelArgs


# ---------------- 注册模型和配置 ----------------
_AVAILABLE_MODELS = {}
current_module = os.path.dirname(__file__)
for _, name, _ in pkgutil.iter_modules([current_module]):
    if name != 'basemodel':  # 不要导入 basemodel 本身来查找模型
        module = importlib.import_module(f".{name}", __package__)
        for obj_name, obj in inspect.getmembers(module):  # 获取模块中的所有成员（类、函数等）
            if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
                # 模型类需具有 'model_name' 属性，并且对应的 Args 类名是 Model 类名 + 'Args'
                model_cls = obj
                args_cls_name = model_cls.__name__ + 'Args'
                args_cls = getattr(module, args_cls_name, None)
                model_name_attr = getattr(model_cls, 'model_name', model_cls.__name__.lower()) # 获取模型名
                if args_cls and issubclass(args_cls, BaseModelArgs):
                   _AVAILABLE_MODELS[model_name_attr] = (model_cls, args_cls)


# ---------------- 公开的 API 函数 ----------------
__all__ = ['list_models', 'get_model_and_args']  # 明确暴露公共接口，隐藏内部变量

def list_models() -> list[str]:
    """
    返回一个包含所有已注册模型名称的列表。

    Returns:
        list[str]: 模型名称列表。
    """
    return list(_AVAILABLE_MODELS.keys())


def get_model_and_args(model_name: str) -> Tuple[Type[BaseModel], Type[BaseModelArgs]]:
    """
    根据提供的模型名称，返回对应的模型类和配置类。

    Args:
        model_name (str): 模型的唯一标识符 (在模型类中定义的 model_name)。

    Returns:
        包含模型类和配置类的元组 (ModelClass, ArgsClass)
    """
    if model_name not in _AVAILABLE_MODELS:
        available = list_models()
        raise ValueError(f"未知的模型名称: '{model_name}'. 可用的模型有: {available}")

    return _AVAILABLE_MODELS[model_name]


