for ep_num in range(10):
    print(ep_num)
    # for sample in range(NUM_SAMPLES_EPSD):
    filename = 'NSFNET-demand' + str(ep_num) + '.txt'
    with open(filename, 'r') as f:
        ff = open('NSFNET-realdemand' + str(ep_num) + '.txt', 'a')
        while True:
            lines = f.readline()  # 整行读取数据
            if not lines:
                break

            p_tmp, E_tmp, D_tmp = [int(i) for i in
                                   lines.split(',')]

            ff.write(str(p_tmp-1))
            ff.write(str(','))
            ff.write(str(E_tmp-1))
            ff.write(str(','))
            ff.write(str(D_tmp))
            ff.write('\n')
        ff.close()
