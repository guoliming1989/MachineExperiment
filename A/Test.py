"""
通俗的理解__name__ == '__main__'：假如你叫小明.py，在朋友眼中，你是小明(__name__ == '小明')；
在你自己眼中，你是你自己(__name__ == '__main__')。
if __name__ == '__main__'的意思是：当.py文件被直接运行时，if __name__ == '__main__'之下的代码块将被运行；
当.py文件以模块形式被导入时，if __name__ == '__main__'之下的代码块不被运行。
"""
PI=3.14
def main():
    print("pi:" ,PI)
if __name__ == "__main__":
    main()