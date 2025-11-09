import os
from PIL import Image

if __name__=='__main__':
    # 获取目录下文件名
    file_full_path = os.path.dirname(os.path.abspath(__file__))
    files = os.listdir(file_full_path)
    # 图标大小
    size = (256, 256)

    for inName in files:
        # 分离文件名与扩展名
        tmp = os.path.splitext(inName)
        # 因为python文件跟图片在同目录，所以需要判断一下
        if tmp[1] == '.png':
            outName = tmp[0] + '.ico'
            # 打开图片并设置大小
            im = Image.open(os.path.join(file_full_path, inName)).resize(size)
            try:
                # 图标文件保存
                im.save(os.path.join(file_full_path, outName))
                print('{} --> {}'.format(inName, outName))
            except IOError:
                print('connot convert :', inName)