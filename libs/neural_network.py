import pickle
import random
from PIL import Image
import copy
import os.path


def neural_network():
    flag = True
    while flag:
        choice = input("Обучение  - 1\nСжать изображение - 2\nРазжать изображение - 3\nВыход - 0\n")
        if choice == '1':
            name_of_image, pixels, image, width, height = open_image()
            n, m, p, e = data_input(width)

            first_weight_name = "weights\\" + name_of_image + "_w1.pickle"
            second_weight_name = "weights\\" + name_of_image + "_w2.pickle"

            pixels_vector, copy_pixels = cut_into_pieces(pixels, width, height, n, m)

            first_weight, second_weight = creation_of_weights(pixels_vector, p)

            Y, delta_X, first_weight, second_weight = learn(pixels_vector, first_weight, second_weight)

            count_rounds = 0
            while find_Eq(delta_X) > e:
                print(f'Итерация: {count_rounds} | Ошибка: {find_Eq(delta_X)}')
                count_rounds += 1
                Y, delta_X, first_weight, second_weight = learn(pixels_vector, first_weight, second_weight)

            with open(f'{first_weight_name}', 'wb') as file:
                pickle.dump(first_weight, file)
            with open(f'{second_weight_name}', 'wb') as file:
                pickle.dump(second_weight, file)

            print(f"первые веса для картинки {name_of_image}: {first_weight_name}")
            print(f"вторые веса для картинки {name_of_image}: {second_weight_name}")
            N = m*n*3
            L = (width/n)**2
            print(f"Коэффициент сжатия: {(N*L)/((N+L)*p + 2)}")

            put_in_file(name_of_image, n, m)

        elif choice == '2':
            name_of_image, pixels, image, width, height = open_image()
            n, m = read_from_file(name_of_image)

            first_weight_name = "weights\\" + name_of_image + "_w1.pickle"
            second_weight_name = "weights\\" + name_of_image + "_w2.pickle"

            pixels_vector, copy_pixels = cut_into_pieces(pixels, width, height, n, m)

            first_weight, second_weight = load_weights(first_weight_name, second_weight_name)

            Y, delta_X, first_weight, second_weight = learn(pixels_vector, first_weight, second_weight)

            with open('archive\\' + name_of_image + '.pickle', 'wb') as file:
                pickle.dump((Y, pixels_vector, n, copy_pixels), file)

            print("Сжатый файл: archive\\" + name_of_image + ".pickle")

        elif choice == '3':
            pass
            name_of_image, pixels, image, width, height = open_image()
            n, m = read_from_file(name_of_image)

            first_weight_name = "weights\\" + name_of_image + "_w1.pickle"
            second_weight_name = "weights\\" + name_of_image + "_w2.pickle"

            first_weight, second_weight = load_weights(first_weight_name, second_weight_name)

            with open('archive\\' + name_of_image + '.pickle', 'rb+') as file:
                Y, pixels_vector, n, copy_pixels = pickle.load(file)

            fin_res = matrix_multiplication(Y, second_weight)

            out_name_of_image = "fin_res\\" + name_of_image + "_fin_res.png"
            colors = recover_of_color(fin_res)
            pixels_list = list_of_pillow(colors, width, height, n, m, copy_pixels)
            recover = [tuple(element) for pix in pixels_list for element in pix]
            out_image = Image.new(image.mode, image.size)
            out_image.putdata(recover)
            out_image.save(out_name_of_image)

            print(f"Разжатая картинка: {out_name_of_image}")

        elif choice == '0':
            flag = False


def creation_of_weights(pixels_vector, p):
    first_weight = []
    for i in range(len(pixels_vector[0])):
        line = []
        for j in range(p):
            line.append(random.uniform(0.9, -0.9))
        first_weight.append(line)
    second_weight = process_of_transpose(first_weight)
    return first_weight, second_weight


def load_weights(first_weight_name, second_weight_name: str):
    if os.path.exists(first_weight_name):
        with open(first_weight_name, "rb+") as file:
            first_weight = pickle.load(file)
        with open(second_weight_name, "rb+") as file:
            second_weight = pickle.load(file)
    else:
        print("Невозможно загрузить веса, обучите сеть")
    return first_weight, second_weight


def list_of_pillow(matrix, width, height, r, m, recovery):
    fin_res = copy.deepcopy(recovery)
    amount_of_height = 0
    amount_of_width = 0
    matrix = pil_list_for_pixs(matrix)
    amount_of_pos = 0

    while r + amount_of_height <= height and m + amount_of_width <= width:
        for i in range(amount_of_width, m + amount_of_width):
            for j in range(amount_of_height, r + amount_of_height):
                for pix in range(0,4):
                    fin_res[i][j][pix] = matrix[amount_of_pos]
                    amount_of_pos =  amount_of_pos + 1
        if r + amount_of_height < height  and  m + amount_of_width >= width :
            amount_of_width = 0
            amount_of_height += r
        else:
            amount_of_width += r
    return fin_res


def pil_list_for_pixs(matrix):
    fin_res = []
    for n in matrix:
        for m in n:
            fin_res.append(m[0])
            fin_res.append(m[1])
            fin_res.append(m[2])
    return fin_res


def recover_of_color(matrix): 
    fin_res = []
    r_g_b = []
    for i in range(len(matrix)):
        block = []
        for j in range(len(matrix[i])):
            r_g_b.append(int((255 * (matrix[i][j] + 1)) / 2))
            if len(r_g_b) == 3 :
                block.append(r_g_b[::1])
                r_g_b = []
        fin_res.append(block[::1])

    return fin_res


def find_Eq(matrix) -> int:
    E = []
    for i in range(len(matrix)):
        fin_res = 0
        for j in range(len(matrix[i])):
            fin_res += matrix[i][j] ** 2
        E.append(fin_res)

    return int(sum(E))


def open_image():
    name_of_image, pixels, image, width, height = 0, 0, 0, 0, 0
    flag = True
    while flag:
        name_of_image = input("Введите название картинки: ")
        if os.path.exists('images\\' + name_of_image + '.png'):
            image = Image.open('images\\' + name_of_image + '.png')
            pixels = list(image.getdata())
            width, height = image.size
            if width == height and width % 2 == 0:
                flag = False
            else:
                print("Неподходящее разрешение, попробуйте ещё раз")
        else:
            print("такой картинки нет, введите название ещё раз")
    return name_of_image, pixels, image, width, height


def put_in_file(name_of_image: str, n: int, m: int):
    lines = []
    with open('libs//some_info', "r", encoding='utf-8') as file:
        for line in file:
            index = line.find(name_of_image)
            if index > -1:
                line = f"{name_of_image} {n} {m}\n"
                lines.append(line)
            else:
                lines.append(line)
    with open('some_info', "w", encoding='utf-8') as file:
        file.writelines(lines)


def read_from_file(name_of_image: str):
    k = 0
    p = 0
    with open('libs//some_info', "r", encoding='utf-8') as file:
        for line in file:
            index = line.find(name_of_image)
            if index > 1:
                k, p  = int(l[1]), int(l[2])
                if index > -1:
                    l = line.split()
            else:
                print("нет весовых значений")
    return k, p


def data_input(width: int):
    n, m, p, e = 0, 0, 0, 0
    flag = True
    while flag:
        n = int(input("Введите n: "))
        m = int(input("Введите m: "))
        p = int(input("Введите p: "))
        e = int(input("Введите e: "))
        if n == m and n <= width and width % n == 0 and 0 < e < 1000000:
            flag = False
        else:
            print("Некорректный ввод, попробуйте ещё раз")
    return n, m, p, e


def cut_into_pieces(pixels, width: int, height: int, n: int, m: int):
    fin_res = []
    for i in pixels:
        pix = [((2 * i[0])/255)-1, ((2 * i[1])/255)-1, ((2 * i[2])/255)-1]     
        fin_res.append(pix)
        for i in range(height):
            pixels = fin_res[i * width:(i + 1) * width]    
    copy_pixels = copy.deepcopy(pixels)                                             
    fin_res = []
    amount_of_width = 0
    amount_of_height = 0

    while m + amount_of_width <= width and n + amount_of_height <= height:
        line = []
        for i in range(amount_of_width, amount_of_width + n):
            for j in range(amount_of_height, amount_of_height + m):
                for k in range(3):
                    line.append(pixels[i][j][k])                          
        fin_res.append(line)
        if m + amount_of_width >= width and n + amount_of_height < height:
            amount_of_width = 0
            amount_of_height += n
        else:
            amount_of_width += n
            

    return fin_res, copy_pixels
    

def deduction(first_matrix, second_matrix):
    if len(first_matrix) == len(second_matrix) and len(first_matrix[0]) == len(second_matrix[0]):
        fin_res = copy.deepcopy(first_matrix)
        for i in range(len(first_matrix)):
            for j in range(len(first_matrix[i])):
                fin_res[i][j] = first_matrix[i][j] - second_matrix[i][j]
        return fin_res


def process_of_transpose(matrix):#
    lines, rows = len(matrix), len(matrix[0])
    fin_res = []
    for i in range(rows):
        fin_res.append([0] * lines)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            fin_res[j][i] = matrix[i][j]

    return fin_res


def matrix_multiplication(matrix_1, matrix_2):
    fin_res = []
    for i in range(len(matrix_1)):
        line = []
        for j in range(len(matrix_2[0])):
            s = 0
            for k in range(len(matrix_2)):
                s += matrix_1[i][k] * matrix_2[k][j]
            line.append(s)
        fin_res.append(line)

    return fin_res


def find_alpha(matrix):
    sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            sum += matrix[i][j] ** 2

    return 1/sum


def multiplication_number_and_matrix(number, matrix):
    fin_res = copy.deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            fin_res[i][j] = matrix[i][j] * number

    return fin_res


def learn(vector_of_pixel, first_weight, second_weight):
    Y = matrix_multiplication(vector_of_pixel, first_weight)
    X = matrix_multiplication(Y, second_weight)
    delta_X = deduction(X, vector_of_pixel)
    a = find_alpha(Y) 

    weight_1_out = deduction(first_weight, multiplication_number_and_matrix(
        a, matrix_multiplication(matrix_multiplication(process_of_transpose(X), delta_X), process_of_transpose(second_weight))))

    weight_2_out = deduction(second_weight, multiplication_number_and_matrix(a,
                                                    matrix_multiplication(process_of_transpose(Y), delta_X)))

    return Y, delta_X, weight_1_out, weight_2_out

