import os

def draw_data_import(filename):
    classes_path = os.path.expanduser(filename)
    with open(classes_path, 'r', encoding='UTF-8') as f:
        class_names = f.readlines()
    class_names = [int(c.strip()) for c in class_names]
    return class_names


