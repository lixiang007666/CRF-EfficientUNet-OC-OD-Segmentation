from PIL import Image
import os.path
import glob


def Resize(file, outdir, width, height):
    imgFile = Image.open(file)
    try:
        newImage = imgFile.resize((width, height), Image.BILINEAR)
        newImage.save(os.path.join(outdir, os.path.basename(file)))
    except Exception as e:
        print(e)


for file in glob.glob("origin/*.png"):  # 图片所在的目录
    Resize(file, "after", 512, 512)  # 新图片存放的目录