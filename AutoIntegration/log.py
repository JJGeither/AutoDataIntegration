# log.py

do_logging = True


def log(str):
    if do_logging:
        print(str)
        with open('data/log.txt', mode='a', newline='') as file:
            file.write(str + "\n")


def set_logging(bool):
    global do_logging
    do_logging = bool
