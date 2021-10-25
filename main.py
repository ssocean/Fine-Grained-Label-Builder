import argparse
import copy
import cv2
import numpy as np
import labelme2coco  # pip install labelme2coco
import json
import glob
import os
from os.path import splitext
from tqdm import tqdm


def auto_make_directory(dir_pth: str):
    '''
    自动检查dir_pth是否存在，若存在，返回真，若不存在创建该路径，并返回假
    :param dir_pth: 路径
    :return: bool
    '''
    if os.path.exists(dir_pth):  ##目录存在，返回为真
        return True
    else:
        os.makedirs(dir_pth)
        return False


def get_files_pth(dir_pth: str, suffix: str = '*'):
    '''
    返回dir_pth下以后缀名suffix结尾的文件绝对路径list
    :param dir_pth:文件夹路径
    :param suffix:限定的文件后缀
    :return: 文件绝对路径list
    '''
    rst = []
    glob_pth = os.path.join(dir_pth, f'*.{suffix}')
    for filename in glob.glob(glob_pth):
        rst.append(filename)
    return rst


#


def get_files_name(dir_path: str, suffix: str = '*'):
    '''
    返回指定文件夹内的文件名（不带后缀）列表
    :param dir_path: 文件夹路径
    :param suffix:限定的文件后缀
    :return:文件名（不带后缀）列表
    '''
    if suffix == '*':
        ids = [splitext(file)[0] for file in os.listdir(dir_path) if not file.startswith('.')]
        return ids
    else:
        ids = [splitext(file)[0] for file in os.listdir(dir_path) if file.endswith(f'.{suffix}')]  # 获取图片名称，ids是一个列表
        return ids


def get_filename_from_pth(file_pth: str, suffix: bool = True):
    '''
    根据文件路径获取文件名
    :param file_pth:文件路径
    :return:文件名
    '''
    fname_list = os.path.split(file_pth)[1].split('.')
    rst = '.'.join(fname_list)
    return rst


def find_contours(rst_img, ori_img, dilate_times=10, erode_times=2):
    '''
    返回img中指定颜色color的轮廓
    :param img:
    :param color:
    :return:
    '''
    h, w, _ = ori_img.shape
    poly_img = copy.deepcopy(ori_img)
    # ---------------------------------------------------------------------------------------------------------更改------------------------------------------------------------------------
    struc_ele = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 创建结构元素

    rst_img = cv2.erode(rst_img, struc_ele,
                        iterations=dilate_times)  # actually it's dilate operation due to the white background
    contourLines = []
    rst_img = cv2.dilate(rst_img, struc_ele, iterations=erode_times)  # erode operation

    rst_img = ~rst_img
    contours, hierarchy = cv2.findContours(rst_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_TC89_L1)  # The last paramenter helps reduce the number of point along the contour.

    for i in range(len(contours)):
        cnt = contours[i]
        if cv2.contourArea(contours[i]) < (
                h * w) / 6000:  # 6000 is based on our erperience, change it to a lower value if the image is smaller.
            # here, 'if' is used to denoise
            continue
        else:
            temp = contours[i]
            epsilon = 0.0001 * cv2.arcLength(cnt, True)  # tune the '0.0001' if the contour looks terrible
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            cv2.polylines(poly_img, [approx], True, (0, 0, 255), 2)
            contourLines.append(temp)
    return poly_img, contourLines


def get_mask_by_color(img: np.ndarray, color: int):
    '''
    get the mask of gray-scale img according to designated color
    :param img: a white background img (in a B,G,R order)
    :param color: intensity
    :return: rst which has the same size as img, while only contain the desired mask
    '''
    img = ~img
    b, g, r = cv2.split(img)
    r[r != (255 - color)] = 0
    color_map = (255 - color) * np.ones(img.shape[:-1], dtype=np.uint8)
    # rst = np.zeros(img.shape[:-1], dtype=np.uint8)
    rst = cv2.bitwise_and(r, color_map)
    rst = (rst / (255 - color)) * 255
    rst = rst.astype(np.uint8)
    rst = ~rst
    return rst


def resize_imgdir(img_dir, rst_dir, size: tuple):
    '''
    This function is used to resize all img in img_dir to the designated size and output the rescaled one to rst_dir
    :param img_dir: image dir
    :param rst_dir: output dir
    :param size:
    :return:
    '''
    img_pths = get_files_pth(img_dir)
    for img_pth in tqdm(img_pths):
        img = cv2.imread(img_pth)
        img = cv2.resize(img, size)  # 2496, 800
        cv2.imwrite(os.path.join(rst_dir, os.path.basename(img_pth)), img)
    pass


class Label:
    '''
    Each instance of Label is an image
    '''

    def __init__(self,
                 abs_img_pth: str,
                 json_pth: str = '',
                 version: str = '4.5.7',
                 flags: dict = {},
                 shapes: list = [],
                 imagePath: str = '',
                 imageData: str = 'None',
                 imageHeight: int = 0,
                 imageWidth: int = 0
                 ):
        '''
        该类可由两种方法初始化，直接加载json与分别指定值
        :param label:
        :param version:
        :param flags:
        :param shapes:
        :param imagePath:
        :param imageData:
        :param imageHeight:
        :param imageWidth:
        '''
        self.output_shape = None
        self.abs_img_pth = abs_img_pth
        self.img = cv2.imread(abs_img_pth)
        self.input_shape = self.img.shape
        if json_pth == '':  # has no corresponding json file
            self.imageWidth = float(self.input_shape[1])
            self.imageHeight = float(self.input_shape[0])
            self.imageData = imageData
            self.imagePath = imagePath
            self.shapes = shapes
            self.flags = flags
            self.version = version
        else:  # load json file
            self.json_pth = json_pth
            with open(json_pth, 'r') as f:
                label = json.loads(f.read())
            self.imageWidth = label['imageWidth']
            self.imageHeight = label['imageHeight']
            self.imageData = label['imageData']
            self.imagePath = label['imagePath']
            self.shapes = label['shapes']
            self.flags = label['flags']
            self.version = label['version']
        # 辅助成员变量

    def __get_final_label__(self):
        rst = {}
        rst["version"] = self.version
        rst['flags'] = self.flags
        rst['shapes'] = self.shapes
        rst['imagePath'] = self.imagePath
        rst['imageData'] = self.imageData
        rst['imageHeight'] = self.imageHeight
        rst['imageWidth'] = self.imageWidth
        return rst

    def resize_(self, rst_shape: tuple, img_relative_pth: str = ''):
        '''
        原地操作函数，由于原图尺寸的变换将会导致标注信息的变换，该方法完成在图片尺寸变换时标注信息的同步转换。
        最好由低分辨率放大至高分辨率
        :param rst_img_pth: 原图变换后图片IMG的路径
        :param img_relative_pth: 原图变换后图片IMG相对于JSON文件的相对路径，该路径仅用于向JSON中imagePath字段写入，默认文文件名
        :return:
        '''
        self.imageData = None  # disable imageData
        self.output_shape = rst_shape

        o_h, o_w, o_c = self.imageHeight, self.imageWidth, rst_shape[2]

        r_h, r_w, r_c = rst_shape
        height_ratio = r_h / o_h
        width_ratio = r_w / o_w  # 计算出高度、宽度的放缩比例

        # 对每个点都乘该比例
        for index, l_dict in enumerate(self.shapes):  # 获取每个标签字典
            # l_dict是一个字典，包含label,points等键
            new_shapes = []
            for dot in l_dict['points']:
                new_shapes.append([dot[0] * width_ratio, dot[1] * height_ratio])
            self.shapes[index]['points'] = new_shapes
        self.imageHeight *= height_ratio
        self.imageWidth *= width_ratio

        if not img_relative_pth == '':  # 如果未提供相对路径,则为文件名
            self.imagePath = img_relative_pth
        return True

    def bin_to_coco(self, img_pth='', d_t=10, e_t=2):
        if img_pth == '':
            img_pth = self.abs_img_pth
        print('img_path:' + img_pth)
        img = cv2.imread(img_pth, 1)
        shapes = []
        for j in range(1, 11):  # edit here to (1,class numer+1)
            shape_dict = {}
            temp_mask = get_mask_by_color(img, j * 20)  # edit here
            _, con_line = find_contours(temp_mask, img, d_t, e_t)
            points = []
            # edit here
            if j <= 8:
                shape_dict['label'] = f'line{j}'
            elif j == 9:
                shape_dict['label'] = f'ltitle'
            elif j == 10:
                shape_dict['label'] = f'rtitle'

            for g_id, point_array in enumerate(con_line):
                shape_dict_deepcopy = copy.deepcopy(shape_dict)
                for point in point_array:
                    # if random.randint(1, 10)<5:
                    points.append([float(point[0][0]), float(point[0][1])])
                shape_dict_deepcopy['points'] = points
                points = []
                shape_dict_deepcopy["group_id"] = str(g_id)

                shape_dict_deepcopy['shape_type'] = 'polygon'
                shape_dict_deepcopy['flags'] = {}
                shapes.append(shape_dict_deepcopy)
                # print(
                #     f'{shape_dict_deepcopy["label"]}下第{shape_dict_deepcopy["group_id"]}个标注生成了，该标注共有{len(point_array)}个点')
        self.shapes = shapes
        # print('done')
        pass

    def output(self, rst_pth: str = '', rst_img_pth: str = ''):
        '''
        输出此对象至rst_pth
        :param rst_pth: 输出路径，默认为原JSON路径
        :return:
        '''
        if rst_pth == '':
            assert self.json_pth != '', 'JSON路径为空，请检查实例对象初始化部分代码'
            rst_pth = self.json_pth
        rst = {}
        rst["version"] = str(self.version)
        rst['flags'] = {}
        rst['shapes'] = self.shapes
        rst['imagePath'] = str(self.imagePath)
        rst['imageData'] = None  # self.imageData
        rst['imageHeight'] = round(float(self.imageHeight), 1)
        rst['imageWidth'] = round(float(self.imageWidth), 1)
        json.dumps(rst)
        with open(rst_pth, 'w+') as f:  # 打开文件用于读写，如果文件存在则打开文件，将原有内容删除；文件不存在则创建文件；
            # pass
            f.write(json.dumps(rst))
        if not rst_img_pth == '':
            img = cv2.imread(self.abs_img_pth)
            h, w, c = self.output_shape

            img = cv2.resize(img, (w, h))
            # print(img.shape)
            cv2.imwrite(rst_img_pth, img)
            return True


def step2_1(source_dir='', rst_dir='', d_t=10, e_t=2):
    '''
    第二步 将文本行灰度图转换为labelme格式
    :param source_dir: 存放灰度文本行的文件夹路径
    :param rst_dir: labelme结果存放文件夹路径
    :return:
    '''
    auto_make_directory(rst_dir)
    source_files = get_files_pth(source_dir)
    for file in tqdm(source_files):
        file_name = os.path.basename(file)
        labelme_obj = Label(abs_img_pth=file, imagePath=file_name)
        labelme_obj.bin_to_coco('', d_t, e_t)
        labelme_obj.resize_((args.height, args.width, 3), img_relative_pth=file_name)
        labelme_obj.output(os.path.join(rst_dir, file_name.split('.')[0] + '.' + file_name.split('.')[1] + '.json'))


def resize_imgs_to(source_dir='', rst_dir=''):
    '''
    resize all imgs in source_dir to rst_dir
    :param source_dir: 原图
    :param rst_dir: resize后的结果图存放路径
    :return:
    '''
    auto_make_directory(rst_dir)
    resize_imgdir(source_dir, rst_dir, (args.height, args.width))


def step_3(labelme_folder='', rst_dir=''):
    '''
    labelme转coco
    :param labelme_folder: 存放labelme文件的文件夹路径
    :param rst_dir: 存放coco文件的文件夹路径
    :return:
    '''

    auto_make_directory(rst_dir)

    # set path for coco json to be saved
    save_json_path = os.path.join(rst_dir, "annotations-origin.json")

    # convert labelme annotations to coco
    labelme2coco.convert(labelme_folder, save_json_path)
    print('Success，Svaing JSONO to:' + r'' + save_json_path)


def step_3_aux(pth):
    '''
    将coco文件的文件路径去除windows系统痕迹，以便将数据用在linux上。
    :param pth: 存放coco文件的路径
    :return:
    '''

    json_pth = os.path.join(pth, "annotations-origin.json")
    with open(json_pth, 'r') as f:
        label = json.loads(f.read())

    images = label['images']
    for image in tqdm(images):
        image['file_name'] = image['file_name'].split('\\')[-1]
    label['images'] = images
    json.dumps(label)
    with open(os.path.join(pth, "annotations.json"), 'w+') as f:  # 打开文件用于读写，如果文件存在则打开文件，将原有内容删除；文件不存在则创建文件；
        # pass
        f.write(json.dumps(label))


def build_dataset_pipeline(base_dir):
    dilate_erode_times = [[6, 4], [10, 2]]
    if not base_dir:
        base_dir = r'Input ur base dir here'

    source_dir = os.path.join(base_dir, r'source_img')
    segmented_dir = os.path.join(base_dir, r'segmented_img')

    out_dir = os.path.join(base_dir, r'Annotations')

    # test 1 rst dir
    for d_e_t in tqdm(dilate_erode_times):
        print('converting segmented_img to labelme' + f'{str(d_e_t[0] - d_e_t[1])}_{str(d_e_t[0])}-{str(d_e_t[1])}')

        labelme_dir = os.path.join(out_dir,
                                   f'{str(d_e_t[0] - d_e_t[1])}_{str(d_e_t[0])}-{str(d_e_t[1])}/labelme')  # test 2 rst dir
        # test 3 source dir
        step2_1(segmented_dir, labelme_dir, d_t=d_e_t[0], e_t=d_e_t[1])  # 生成labelme的JSON值trd2
        resize_imgs_to(source_dir, labelme_dir)  # resize原图至trd2

        print('converting labelme to coco，this may take a few time')
        coco_dir = os.path.join(out_dir,
                                f'{str(d_e_t[0] - d_e_t[1])}_{str(d_e_t[0])}-{str(d_e_t[1])}')  # test 4 rst dir
        step_3(labelme_dir, coco_dir)
        step_3_aux(coco_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("base_dir")
    parser.add_argument("-height", help="resize height", type=int, default=700)
    parser.add_argument("-width", help="resize width", type=int, default=700)
    args = parser.parse_args()
    build_dataset_pipeline(args.base_dir)
