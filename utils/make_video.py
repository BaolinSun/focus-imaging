import cv2
import os
from natsort import natsorted

def images_to_video(image_folder, output_video, fps=30, frame_size=None):
    """
    将文件夹内的图片制作成视频。
    
    :param image_folder: 包含图片的文件夹路径
    :param output_video: 输出视频文件的路径
    :param fps: 视频的帧率
    :param frame_size: 视频帧的大小（宽, 高），默认为None，根据第一张图片自动设置
    """
    # 获取文件夹内所有图片文件并排序
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)  # 自然排序，确保图片按照数字命名顺序排列
    
    if not images:
        print("没有找到图片文件")
        return

    # 获取第一张图片以确定帧的大小
    first_image_path = os.path.join(image_folder, images[0])
    first_frame = cv2.imread(first_image_path)
    if frame_size is None:
        frame_size = (first_frame.shape[1], first_frame.shape[0])  # (宽, 高)

    # 初始化视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    x1, y1, x2, y2 = 30, 20, 226, 256  # 裁剪区域的左上角和右下角坐标

    for i in range(10):
        for image_name in images:
            image_path = os.path.join(image_folder, image_name)
            frame = cv2.imread(image_path)

            # frame = cv2.GaussianBlur(frame, (5, 5), sigmaX=1.0)
            # frame = frame[y1:y2, x1:x2]

            # frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=0)

            # 检查图像大小是否与指定的帧大小一致
            if (frame.shape[1], frame.shape[0]) != frame_size:
                frame = cv2.resize(frame, frame_size)

            out.write(frame)  # 写入帧到视频文件

    out.release()
    print(f"视频已保存到 {output_video}") 


# 示例用法
if __name__ == "__main__":
    image_folder = "usimage"  # 替换为你的图片文件夹路径
    output_video = "usvideo/output28.mp4"          # 输出视频文件名
    fps = 5                                  # 帧率
    images_to_video(image_folder, output_video, fps)
