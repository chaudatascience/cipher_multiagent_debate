def most_frequent(a_list):
    a_list = a_list[::-1]
    counter = 0
    num = a_list[0]

    for i in a_list:
        current_frequency = a_list.count(i)
        if current_frequency > counter:
            counter = current_frequency
            num = i
    return num
