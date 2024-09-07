import click
import pyfiglet
from colorama import Fore
import os
from tqdm import tqdm
import time
from pathlib import Path
import psutil
import logging
import sys
import pandas as pd
from rich.console import Console
from rich.table import Table

import datetime






def to_color(string, color):
    color_code = {'blue': '\033[34m',
                    'yellow': '\033[33m',
                    'green': '\033[32m',
                    'red': '\033[31m'
                    }
    return color_code[color] + str(string) + '\033[0m'


@click.command()
@click.option("--t_mode", prompt="Введите тип процесса - \n1 - predict(предсказание)\n2 - train(обучение)\n3 - inc_learn(дообучение)\nq - выход\n->")
def mode(t_mode):
    print(psutil.virtual_memory().free)
    if t_mode == "predict" or t_mode == "1":
        predict()
    elif t_mode == "train" or t_mode == "2":
        train()
    elif t_mode == "inc_learn" or t_mode=="3":
        inc_learn()
    elif t_mode=="q":
        exit(0)
    else:
        mode()



def replace_df(x):
    if type(x)==int and x<=0:
        return '!!!Возможна скорая поломка!!!'
    else:
        return "{}".format(x)


@click.command()
@click.option("--path", prompt="Введите путь файла, для которого будет происходит предсказание")
def predict(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        print(Fore.GREEN+"Путь существует и есть файл нужного расширения, процесс запущен")
        df = pd.read_csv(path)
        size = df.shape
        df = df.sort_values("Количество часов работы")
        df['Количество часов работы'] = df['Количество часов работы'].apply(replace_df)
        table = Table(title="Disks")
        rows = df.iloc[:15].values.tolist()
        rows = [[str(el) for el in row] for row in rows]
        for column in df.columns:
            table.add_column(column, vertical="middle")
        for row in rows:
            table.add_row(*row,  style='bright_green')
        console = Console()
        console.print(table)
        print(Fore.YELLOW+"Получившиеся значения сохранены в файле result_{}.csv\nВсего количество элементов - {}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M"), size[0]))
        df.to_csv("result_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")), index=False)
        mode()
    else:
        print(Fore.RED+"Файл не того расширения или его не существует")
        print(Fore.YELLOW+"")
        predict()



@click.command()
@click.option("--path", prompt="Введите путь директории, для которого будет происходит обучение")
def train(path):
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files)==0:
            print(Fore.RED+"В директории нет ни одного файла подходящего расширения")
            print(Fore.YELLOW+"")
            train()
        else:
            print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
            for i in tqdm(range(len(files)), file=sys.stdout):
                time.sleep(3)
                tqdm.write("{}, {}".format(5, files[i]))
            print(Fore.YELLOW+"")
            mode()
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")
        print(Fore.YELLOW+"")
        train()
    
        



@click.command()
@click.option("--path", prompt="Введите путь директории, для которого будет происходит дообучение")
def inc_learn(path):
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files)==0:
            print(Fore.RED+"В директории нет ни одного файла подходящего расширения")
            print(Fore.YELLOW+"")
            inc_learn()
        else:
            print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
            for i in tqdm(range(len(files))):
                time.sleep(3)
                tqdm.write("{}, {}".format(5, files[i]))
            print(Fore.YELLOW+"")
            mode()
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")
        print(Fore.YELLOW+"")
        inc_learn()
    
    



def hello():
    mydsa = pyfiglet.figlet_format("MYDSA", font="slant")
    print(to_color(mydsa, "blue"))
    print(Fore.MAGENTA + "MYDSA - Make Your Disk Smart Again\nВерсия утилиты 1000 минус 7")
    print(Fore.YELLOW+"")


if __name__ == "__main__":
    hello()
    mode()

