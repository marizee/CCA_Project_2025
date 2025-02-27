import sys

try:
    file_name = sys.argv[1]
except IndexError:
    print("No input file. Running on scalar-vector_25bits_marie_laptop.txt")
    file_name = "scalar-vector_25bits_marie_laptop.txt"

try:
    datas = open(file_name,"r").read().splitlines()
except FileNotFoundError:
    print("File doesn't exists. Running on scalar-vector_25bits_marie_laptop.txt")
    datas = open("scalar-vector_25bits_marie_laptop.txt", "r").read().splitlines()
    fratios = open("ratio_scalar-vector_25bits_marie_laptop.txt", "w")
else:
    fratios = open("ratio_" + file_name, "w")


parse = datas[5].split()
avx512 = False if len(parse)==6 else True

fratios.write(f'{parse[0]}\t{parse[2]}\t{parse[3]}\t{parse[4]}\t{parse[5]}')
if avx512:
    fratios.write(f'{parse[6]}\t{parse[7]}\n')
else:
    fratios.write("\n")

for row in datas[6:]:
    parse = row.split()
    ratios = [float(parse[1])/float(parse[i]) for i in range(2,len(parse))]
    if len(parse[0]) <= 3:
        fratios.write(f'{parse[0]}\t\t')
    else:
        fratios.write(f'{parse[0]}\t\t')
    fratios.write(f'{ratios[0]:.3f}\t\t{ratios[1]:.3f}\t{ratios[2]:.3f}\t{ratios[3]:.3f}')
    if avx512:
        fratios.write(f'\t{ratios[4]:.3f}\t{ratios[5]:.3f}\n')
    else:
        fratios.write("\n")

fratios.close()
