import sys

try:
    file_name = sys.argv[1]
except IndexError:
    print("Missing parameter.")
    print("Usage: python3 ratio.py [file_timings] [avx512]")
    print("    - avx512: 1 if exists, 0 else")
    exit()

try:
    avx512 = int(sys.argv[2])
except IndexError:
    print("Missing parameter.")
    print("Usage: python3 ratio.py [file_timings] [avx512]")
    print("    - avx512: 1 if exists, 0 else")
    exit()

try:
    datas = open(file_name,"r").read().splitlines()
except FileNotFoundError:
    print("Problem opening file.")
    exit()

h = datas.index('')
parse = datas[h+1].split()
nbv = len(parse)-1

fratios = open("ratio_" + file_name, "w")
fratios.write(f'{parse[0]}\t')
for i in range(2,nbv+1):
    fratios.write(f'{parse[i]}\t\t')
fratios.write("\n")

for row in datas[h+2:]:
    parse = row.split()

    ratios = [float(parse[1])/float(parse[i]) for i in range(2,len(parse))]
    if len(parse[0]) <= 3:
        fratios.write(f'{parse[0]}\t\t\t')
    else:
        fratios.write(f'{parse[0]}\t\t')
    
    for r in ratios:
        fratios.write(f'{r:.3f}\t\t')
    fratios.write("\n")

fratios.close()
