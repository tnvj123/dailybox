import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))
print(os.getcwd())
image_directory = os.path.join(os.getcwd(), 'data', 'validation')
print(f"Image directory: {image_directory}")
for class_dir in os.listdir(image_directory):
    class_path = os.path.join(image_directory)
    if os.path.isdir(class_path):
        print(f"Files in {os.path.abspath(class_path)}: {os.listdir(class_path)}")

def 젠장에이스이공격은대체뭐냐(n):
    if n == 1:
        return "젠장에이스공격은대체뭐냐!"
    elif n == 2:
        return "몸이점점달아오르잔아!"
    elif n == 3:
        return "젠장티치!난니가좋다!"
    else:
        return "드디어왔다..조이는보이가!"