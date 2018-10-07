import tifffile as tiff
import os

"""
2016年的原图是4通道的，而且蓝绿通道顺序反了

处理图像：将蓝绿通道交换顺序，并去掉红外波段

"""

def RGBA_RGB(path, save_path):
    fileList = os.listdir(path)
    for file in fileList:
        image = tiff.imread(os.path.join(path, file))
        # 交换通道
        image = image[:, :, (2, 1, 0)]
        # image = image.resize((1000, 1000))
        print(image.shape)
        tiff.imsave(save_path + file, image)


# 只保留后缀是.TIF的图像
def remove_file(path):
    fileList = os.listdir(path)
    for file in fileList:
        pathname = os.path.splitext(os.path.join(path, file))
        if pathname[1] != ".TIF":
            os.remove(os.path.join(path, file))


def zh_ch(string):
    return string.encode("gbk").decode(errors="ignore")


if __name__ == '__main__':
    # path = r'F:\remote_sensing\16年原图\\'
    # save_path = r'F:\remote_sensing\16年_RGB\\'

    path = r'F:\remote_sensing\2017_10\process_1\\10_water\\'
    save_path = r'F:\remote_sensing\2017_10\process_channel_change\\10_water\\'
    # remove_file(path)

    # RGBA_RGB(path, save_path)
    """
    原因分析：这是由于在OpenCV-Python包中，
    imshow函数的窗口标题是gbk编码，而Python3默认UTF-8编码。因而窗口标题包含中文时，会显示乱码。
    """
    import cv2
    im = cv2.imread(r'F:\remote_sensing\2017_10\process_channel_change\01_gengdi\example_img_03_201702_gengdi_10.TIF')

    im_resise = cv2.resize(im, (400, 400), interpolation=cv2.INTER_CUBIC)
    cv2.imshow(zh_ch("Original image size: 458*339"), im)
    cv2.imshow("Scaled size: 400*400", im_resise)
    cv2.waitKey(0)