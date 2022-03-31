from config import *
import os
import cv2 as cv
import numpy as np
import os

class Identifier:

    def __init__(self):
        self.classic_digits = None
        self.jigsaw_grayscale_digits = None
        self.jigsaw_colored_digits = None
    
    
    def __load_templates(self, path, show=False):
        digits = {}
        for i in range(1, 10):
            img = cv.imread(os.path.join(path, str(i) + '.jpg'))
            print(os.path.join(path, str(i) + '.jpg'))
            if show:
                self.__show_image(str(i), img)
            digits[i] = img.astype(np.uint8)
        return digits


    def __get_templates(self, type, is_colored=False):
        if type == CLASSIC:
            if self.classic_digits is None:
                self.classic_digits = self.__load_templates(os.path.abspath(PATH_TEMPLATES_CLASSIC))
            return self.classic_digits
        elif type == JIGSAW:
            if is_colored:
                if self.jigsaw_colored_digits is None:
                    self.jigsaw_colored_digits = self.__load_templates(os.path.abspath(PATH_TEMPLATES_JIGSAW_COLORED))
                return self.jigsaw_colored_digits
            else:
                if self.jigsaw_grayscale_digits is None:
                    self.jigsaw_grayscale_digits = self.__load_templates(os.path.abspath(PATH_TEMPLATES_JIGSAW_GRAYSCALE))
                return self.jigsaw_grayscale_digits
            

    def __show_image(self, title, image):
        cv.imshow(title,image)
        cv.waitKey(0)
        cv.destroyAllWindows()

    
    def __get_corners(self, image, show=False):
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        image_m_blur = cv.medianBlur(image, 3)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), 7) 
        image_sharpened = cv.addWeighted(image_m_blur, 1, image_g_blur, -0.8, 0)
        _, thresh = cv.threshold(image_sharpened, 0, 255, cv.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.erode(thresh, kernel)
        if show: 
            self.__show_image("median blurred",image_m_blur)
            self.__show_image("gaussian blurred",image_g_blur)
            self.__show_image("sharpened",image_sharpened)    
            self.__show_image("threshold of blur",thresh)
        
        edges =  cv.Canny(thresh ,150,400)
        contours, _ = cv.findContours(edges,  cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_area = 0
        
        for i in range(len(contours)):
            if(len(contours[i]) > 2):
                possible_top_left = None
                possible_bottom_right = None
                for point in contours[i].squeeze():
                    if possible_top_left is None or point[0] + point[1] < possible_top_left[0] + possible_top_left[1]:
                        possible_top_left = point

                    if possible_bottom_right is None or point[0] + point[1] > possible_bottom_right[0] + possible_bottom_right[1] :
                        possible_bottom_right = point

                diff = np.diff(contours[i].squeeze(), axis = 1)
                possible_top_right = contours[i].squeeze()[np.argmin(diff)]
                possible_bottom_left = contours[i].squeeze()[np.argmax(diff)]
                if cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]])) > max_area:
                    max_area = cv.contourArea(np.array([[possible_top_left],[possible_top_right],[possible_bottom_right],[possible_bottom_left]]))
                    top_left = possible_top_left
                    bottom_right = possible_bottom_right
                    top_right = possible_top_right
                    bottom_left = possible_bottom_left

        width = 500
        height = 500
        
        image_copy = cv.cvtColor(image.copy(),cv.COLOR_GRAY2BGR)
        cv.circle(image_copy,tuple(top_left),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(top_right),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_left),4,(0,0,255),-1)
        cv.circle(image_copy,tuple(bottom_right),4,(0,0,255),-1)
        if show:
            self.__show_image("detected corners",image_copy)
    
        return top_left,top_right,bottom_left,bottom_right
    

    def __normalize(self, points):
        for point in points:
            point = (point[1], point[0])

        return points


    def __crop_image(self, img, top_left, top_right, bottom_left, bottom_right, show=False):
        ows,cols,ch = img.shape
        top_left, bottom_left, top_right, bottom_right = self.__normalize((top_left, bottom_left, top_right, bottom_right))
        original_points = np.float32([top_left, bottom_left, top_right, bottom_right])
        desired_points = np.float32([[0,0],[0,510],[510,0],[510,510]])
        perspective = cv.getPerspectiveTransform(original_points, desired_points)
        img_crop = cv.warpPerspective(img, perspective, (510,510))
        img_crop = img_crop[3:-3, 3:-3]
        if show:
            self.__show_image('cropped img', img_crop)
        return img_crop


    def __get_lines(self, img_crop, show=False):
        lines_vertical=[]
        img_crop_width = img_crop.shape[1]
        img_crop_height = img_crop.shape[0]
        cell_width = img_crop_width//9
        for i in range(0,img_crop_width, cell_width):
            l=[]
            l.append((i,0))
            l.append((i,img_crop_height))
            lines_vertical.append(l)

        if len(lines_vertical) < 10:
            l = []
            l.append((img_crop_width, 0))
            l.append((img_crop_width, img_crop_height))
            lines_vertical.append(l)

        lines_horizontal=[]
        cell_height = img_crop_height//9
        for i in range(0,img_crop_height,cell_height):
            l=[]
            l.append((0,i))
            l.append((img_crop_width,i))
            lines_horizontal.append(l)
        
        if len(lines_horizontal) < 10:
            l = []
            l.append((0, img_crop_height))
            l.append((img_crop_width, img_crop_height))
            lines_horizontal.append(l)

        return lines_vertical, lines_horizontal


    def __get_results(self, img_crop,lines_horizontal,lines_vertical, show=False, padding_ratio=None):
        patches = []
        padding_height = 0 # img_crop.shape[0] // 90
        padding_width = 0 # img_crop.shape[1] // 90
        
        if padding_ratio is not None:
            padding_height = img_crop.shape[0] // padding_ratio
            padding_width = img_crop.shape[1] // padding_ratio

        for i in range(9):
            for j in range(9):
                y_min = lines_vertical[j][0][0] + padding_height
                y_max = lines_vertical[j + 1][1][0] - padding_height
                x_min = lines_horizontal[i][0][1] + padding_width
                x_max = lines_horizontal[i + 1][1][1] - padding_width
                patch = img_crop[x_min:x_max, y_min:y_max].copy()
                patches.append(patch)
                if show:
                    self.__show_image("patch" + str(i) + ' ' + str(j), patch)
        return patches


    def __is_colored(self, img, color_threshold=COLOR_THRESHOLD):
        B, G, R = (0, 0, 0)
        for line in img:
            for pixel in line:
                b, g, r = pixel
                B += b
                G += g
                R += r

        total_nr_pixels = img.shape[0] * img.shape[1]
        B /= total_nr_pixels
        G /= total_nr_pixels
        R /= total_nr_pixels
        
        if max(B, G, R) - min(B, G, R) > color_threshold:
            return True
        else:
            return False


    def __get_min_chunk(self, patch, coords, chunk_size=CHUNK_SIZE, show=False):
        min_chunk = np.full((chunk_size, chunk_size, 3), 255)
        for i in range(0, patch.shape[0] - chunk_size - 1):
            for j in range(0, patch.shape[1] - chunk_size - 1):
                chunk = patch[i:(i + chunk_size), j:(j + chunk_size)]
                if np.mean(min_chunk) > np.mean(chunk):
                    min_chunk = chunk.copy()
        if show:
            self.__show_image('min chunk '+ str(coords) + ' ' + str(np.mean(min_chunk)), min_chunk)
        return min_chunk


    def __get_cells_map(self, patches, chunk_size=CHUNK_SIZE, digits=None):
        cells_map = np.full((9, 9), 0)
        for i in range(9):
            for j in range(9):
                patch = patches[i*9 + j]
                patch_gs = cv.cvtColor(patch,cv.COLOR_BGR2GRAY)
                padding_height = CROP_HEIGHT // PADDING_RATIO
                padding_width = CROP_WIDTH // PADDING_RATIO
                patch_wp = patch_gs[padding_height:-padding_height, padding_width:-padding_width]
                diff = np.mean(self.__get_min_chunk(patch_wp, (i, j), chunk_size=chunk_size))
                if diff < CHUNK_MEAN_THRESHOLD:
                    if digits is not None:
                        max_matching_value = 0
                        digit = 0
                        for d in range(1, 10):
                            
                            template = cv.cvtColor(digits[d],cv.COLOR_BGR2GRAY)
                            _, patch_gs = cv.threshold(patch_gs, 127, 255, cv.THRESH_BINARY)
                            _, template = cv.threshold(template, 127, 255, cv.THRESH_BINARY)

                            # show_image('template', template)
                            # show_image('patch', patch_gs)
                            # show_image('patch', patch_gs)
                            matching = cv.matchTemplate(patch_gs, template, cv.TM_CCORR_NORMED)
                            if max_matching_value < np.max(matching):
                                max_matching_value = np.max(matching)
                                digit = d

                        cells_map[i, j] = digit
                    else:
                        cells_map[i, j] = 1
            
        return cells_map


    def __compose_answer(self, cells_map, scheme=None, bonus=False):
        answer = ''
        for i in range(9):
            for j in range(9):
                if scheme is not None:
                    answer += str(scheme[i, j])
                if cells_map[i, j] == 0:
                    answer += 'o'
                else:
                    if bonus:
                        answer += str(cells_map[i, j])
                    else:
                        answer += 'x'
            if i < 8:
                answer += '\n'
        return answer
    

    def __get_contours(self, img_crop, jigsaw_type=COLORED, show=False):   
        img_crop = cv.cvtColor(img_crop,cv.COLOR_BGR2GRAY)
        if jigsaw_type == COLORED:
            med_bl = MEDIAN_BLUR_COLORED
            gaus_bl = GAUSS_BLUR_COLORED
        else:
            med_bl = MEDIAN_BLUR_GRAYSCALE
            gaus_bl = GAUSS_BLUR_GRAYSCALE
        image_m_blur = cv.medianBlur(img_crop, med_bl)
        image_g_blur = cv.GaussianBlur(image_m_blur, (0, 0), gaus_bl)
        image_sharpened = cv.addWeighted(image_m_blur, 1, image_g_blur, -0.8, 0)
        _, thresh = cv.threshold(image_sharpened, 0, 255, cv.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv.erode(thresh, kernel)
        if show:
            self.__show_image("median blurred",image_m_blur)
            self.__show_image("gaussian blurred",image_g_blur)
            self.__show_image("sharpened",image_sharpened)    
            self.__show_image("threshold of blur",thresh)
        
        edges =  cv.Canny(thresh ,150,400)
        contours, _ = cv.findContours(edges,  cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        if show:
            res_img = cv.drawContours(img_crop, contours, -1, (255 ,255,255), thickness=1)
            # cv.fillPoly(res_img, pts =contours, color=(255,255,255))
            # contours = np.vstack(contours).squeeze()
            # for p in contours:
            #     img_crop[p[1], p[0]] = 255
            self.__show_image('contours', res_img)
        return contours

    
    def __get_bolds_map(self, contours, jigsaw_type=COLORED):
        if jigsaw_type == COLORED:
            bold_threshold = BOLD_THRESHOLD_COLORED
        else:
            bold_threshold = BOLD_THRESHOLD_GRAYSCALE
        vertical_bolds = np.zeros((9, 10))
        horizontal_bolds = np.zeros((10, 9))

        ## vertical
        for i in range(9):
            for j in range(10):
                if j == 0 or j == 9:
                    vertical_bolds[i][j] = 1
                    continue

                x = j * 56
                y = i * 56 + 28
                min_dist = 10000
                for p in contours:
                    dist = abs(cv.pointPolygonTest(p, (x, y), True))
                    min_dist = min(min_dist, dist)
                if(min_dist < bold_threshold):
                    vertical_bolds[i][j] = 1
        
        # horizontal
        for i in range(10):
            for j in range(9):
                if i == 0 or i == 10:
                    horizontal_bolds[i][j] = 1
                    continue
                x = j * 56 + 28
                y = i * 56
                min_dist = 10000
                for p in contours:
                    dist = abs(cv.pointPolygonTest(p, (x, y), True))
                    min_dist = min(min_dist, dist)
                if(min_dist < bold_threshold):
                    horizontal_bolds[i][j] = 1
        
        return vertical_bolds, horizontal_bolds


    def __get_scheme(self, vertical_bolds, horizontal_bolds):
        harta = []
        for i in range(9):
            line = [False for j in range(9)]
            hline = [0 for j in range(9)]
            harta.append(hline)

        col = [0, -1, 0, 1]
        row = [-1, 0, 1, 0]

        def bfs(harta, i, j, value):
            queue = []
            harta[i][j] = value
            queue.append((i, j))
            while len(queue) > 0:
                i, j = queue[-1]
                queue.pop(-1)
                for d in range(4):
                    ii, jj = i + row[d], j + col[d]
                    
                    if 9 > ii >= 0 and  9 > jj >= 0 and harta[ii][jj] == 0:
                    
                        if ii == i:
                            if jj == j + 1:
                                if vertical_bolds[i][jj] == 0:
                                    harta[ii][jj] = value
                                    queue.append((ii, jj))
                            else:
                                if vertical_bolds[i][j] == 0:
                                    harta[ii][jj] = value
                                    queue.append((ii, jj))
                        else:
                            if ii == i+ 1:
                                if horizontal_bolds[ii][j] == 0:
                                    harta[ii][jj] = value
                                    queue.append((ii, jj))
                            else:
                                if horizontal_bolds[i][j] == 0:
                                    harta[ii][jj] = value
                                    queue.append((ii, jj))
                

        value = 1
        for i in range(9):
            for j in range(9):
                if harta[i][j] == 0:
                    bfs(harta, i, j, value)
                    value += 1

        return np.array(harta)


    def process_image(self, nr, type=CLASSIC, bonus=True, show=False):
        nr_string = str(nr)
        nr_string_for_image = nr_string
        if nr < 10:
            nr_string_for_image = '0' + nr_string
        img = cv.imread(os.path.join(os.path.abspath(PATH_TESTS), type, nr_string_for_image + '.jpg'))
        img = cv.resize(img,(0,0),fx=0.2,fy=0.2)
        top_left,top_right,bottom_left,bottom_right = self.__get_corners(img, show=show)
        if show:
            print('corners:', top_left,top_right,bottom_left,bottom_right)
        
        img_crop = self.__crop_image(img, top_left,top_right,bottom_left,bottom_right, show=show)
        if show:
            print('image cropped')

        lines_vertical, lines_horizontal = self.__get_lines(img_crop)
        if show:
            print('line made')
            print(len(lines_vertical), len(lines_horizontal))

        if type == CLASSIC:
            patches = self.__get_results(img_crop, lines_horizontal, lines_vertical)
        else:
            patches = self.__get_results(img_crop, lines_horizontal, lines_vertical, padding_ratio=PADDING_PATCHES_JIGSAW)
        if show:
            print('patches gotten')

        is_colored = self.__is_colored(patches[0])

        if bonus:
            if type == CLASSIC:
                digits = self.__get_templates(CLASSIC)
            else:
                if is_colored:
                    digits = self.__get_templates(JIGSAW, is_colored=True)
                else:
                    digits = self.__get_templates(JIGSAW)
                
            cells_map = self.__get_cells_map(patches, digits=digits, chunk_size=10)
        else:
            cells_map = self.__get_cells_map(patches, chunk_size=10)
        if show:
            print('cells map\n', cells_map)

        # if bonus:
        #     solution_file = open(os.path.join(PATH_SOLUTIONS, type, nr_string + "_bonus_gt.txt")
        # else:
        #     truth_file = open(path_thruths + type + '/' + nr_string_for_image + "_gt.txt")
        # truth = truth_file.readlines()
        # truth_file.close()
        # if show:
        #     print(truth)

        if type == 'clasic':
            answer = self.__compose_answer(cells_map, bonus=bonus)
        else:
            if is_colored:
                jigsaw_type = COLORED
            else:
                jigsaw_type = GRAYSCALE
            inner_contours = self.__get_contours(img_crop, jigsaw_type=jigsaw_type, show=show)
            vertical_bolds, horizontal_bolds = self.__get_bolds_map(inner_contours, jigsaw_type=jigsaw_type)
            scheme = self.__get_scheme(vertical_bolds, horizontal_bolds)
            if show:
                print('scheme\n', scheme)
            answer = self.__compose_answer(cells_map, scheme, bonus=bonus)

        return answer



            
