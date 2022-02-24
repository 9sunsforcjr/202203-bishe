
filename = 'NSFNET-demand0.txt'
SRC = []
DST = []

with open(filename, 'r') as f:
  while True:
    lines = f.readline() # 整行读取数据
    if not lines:
      break

    p_tmp, E_tmp = [int(i) for i in lines.split(',')] # 将整行数据分割处理，如果分割符是空格，括号里就不用传入参数，如果是逗号， 则传入‘，'字符。
    SRC.append(p_tmp)  # 添加新读取的数据
    DST.append(E_tmp)


print(SRC)
