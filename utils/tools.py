def getClassByName(class_name):
    # 使用内置的 __import__ 函数导入模块
    module = __import__(class_name)
    # 使用内置的 getattr 函数获取类对象
    # my_class = getattr(module, class_name)
    my_class = getattr(module, class_name)
    return my_class