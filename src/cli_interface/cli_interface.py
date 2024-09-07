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
import argparse
from src.model import ModelInterface

#Name of the predicted date column
time_column = 'failure_date'


#Function to add color to logo
def to_color(string, color):
    color_code = {'blue': '\033[34m',
                    'yellow': '\033[33m',
                    'green': '\033[32m',
                    'red': '\033[31m'
                    }
    return color_code[color] + str(string) + '\033[0m'

#Function using menu to select mode
@click.command()
@click.option("--t_mode", prompt="Введите тип процесса - \n1 - predict(предсказание)\n2 - train(обучение)\n3 - inc_learn(дообучение)\nq - выход\n->")
def mode(t_mode):
    # print(psutil.virtual_memory().free)
    if t_mode == "predict" or t_mode == "1":
        predict()
    elif t_mode == "train" or t_mode == "2":
        train()
    elif t_mode == "inc_learn" or t_mode=="3":
        inc_learn()
    elif t_mode=="q":
        raise SystemExit()
    else:
        print('Не удалось распознать ввод\n')
        mode()


#Function of replacing values ​​in a column according to the predicted date
def replace_df(delta):
    if delta < datetime.timedelta(days=90):
        return 'Возможна поломка в течение ближайших трёх месяцев!!!'
    elif delta < datetime.timedelta(days=180):
        return 'Возможна поломка в сроке от трёх до шести месяцев!!'
    elif delta < datetime.timedelta(days=270):
        return 'Возможна поломка в сроке от шести до девяти месяцев!'
    elif delta < datetime.timedelta(days=365):
        return 'Возможна поломка в сроке от девяти до двенадцати месяцев'
    else:
        return "В течение года поломка не предвидется"

#Function for accessing the model for prediction purposes (using the menu)
@click.command()
@click.option("--path", prompt="Введите путь файла, для которого будет происходит предсказание")
def predict(path):
    if os.path.isfile(path) and path.endswith(".csv"):
        if os.path.isfile('model/xgb.bin') and os.path.isfile('model/features.bin') and \
           os.path.isfile('model/label.bin') and os.path.isfile('model/target.bin'):
            model = ModelInterface('model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
            try:
                df = model.predict(path)
            except ValueError:
                print(Fore.RED + "CSV файл не подходящего формата")
            print(Fore.GREEN+"Путь существует и есть файл нужного расширения, процесс запущен")
            size = df.shape
            df = df.sort_values(time_column)
            df['time_delta'] = df[time_column] - datetime.datetime.strptime(os.path.basename(path).replace('.csv', ''), '%Y-%m-%d')
            df['time_delta'] = df['time_delta'].apply(replace_df)
            table = Table(title="Disks")
            rows = df.iloc[:15].values.tolist()
            rows = [[str(el) for el in row] for row in rows]
            for column in df.columns:
                table.add_column(column, vertical="middle")
            for row in rows:
                table.add_row(*row,  style='bright_green')
            console = Console()
            console.print(table)
            print(Fore.YELLOW+"Получившиеся значения сохранены в файле result_{}.csv\nВсего количество элементов - {}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), size[0]))
            df.to_csv("result_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")), index=False)
        else:
            print(Fore.RED + "Модель не существует, нельзя выполнить предсказание. Сначала выполните обучение")
        print(Fore.YELLOW+"")
        mode()
    else:
        print(Fore.RED+"Файл не того расширения или его не существует")
        print(Fore.YELLOW+"")
        predict()


#Function for accessing the model for training purposes (using the menu)
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
            if not os.path.isdir('model'):
                os.mkdir('model')
            model = ModelInterface()
            files = [os.path.join(path, file) for file in files]
            print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
            r2_score, mse = model.train(files, 'model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
            print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            print(Fore.YELLOW+"")
            mode()
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")
        print(Fore.YELLOW+"")
        train()
    
        


#Function for accessing the model for the purpose of additional training (using the menu)
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
            if os.path.isfile('model/xgb.bin') and os.path.isfile('model/features.bin') and \
               os.path.isfile('model/label.bin') and os.path.isfile('model/target.bin'):
                model = ModelInterface('model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
                files = [os.path.join(path, file) for file in files]
                print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
                r2_score, mse = model.inc_train(files)
                print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            else:
                print(Fore.RED + "Модель не существует, нельзя выполнить дообучение. Сначала выполните обучение")
            print(Fore.YELLOW+"")
            mode()
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")
        print(Fore.YELLOW+"")
        inc_learn()
    
#Function for calling the model for prediction purposes (using run arguments)
def arg_predict(path, unsort):
    if os.path.isfile(path) and path.endswith(".csv"):
        if os.path.isfile('model/xgb.bin') and os.path.isfile('model/features.bin') and \
           os.path.isfile('model/label.bin') and os.path.isfile('model/target.bin'):
            model = ModelInterface('model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
            try:
                df = model.predict(path)
            except ValueError:
                print(Fore.RED + "CSV файл не подходящего формата")
            size = df.shape
            if unsort:
                df = df.sort_values(time_column)
            df['time_delta'] = df[time_column] - datetime.datetime.strptime(os.path.basename(path).replace('.csv', ''), '%Y-%m-%d')
            df['time_delta'] = df['time_delta'].apply(replace_df)
            print(Fore.YELLOW+"Получившиеся значения сохранены в файле result_{}.csv\nВсего количество элементов - {}".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"), size[0]))
            df.to_csv("result_{}.csv".format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")), index=False)
    else:
        print(Fore.RED+"Файл не того расширения или его не существует")

#Function for accessing the model for training purposes (using launch arguments)
def arg_train(path):
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files)==0:
            print(Fore.RED+"В директории нет ни одного файла подходящего расширения")
        else:
            if not os.path.isdir('model'):
                os.mkdir('model')
            model = ModelInterface()
            files = [os.path.join(path, file) for file in files]
            print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
            r2_score, mse = model.train(files, 'model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
            print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")

#Function for accessing the model for additional training (using launch arguments)
def arg_inc_learn(path):
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files)==0:
            print(Fore.RED+"В директории нет ни одного файла подходящего расширения")
        else:
            if os.path.isfile('model/xgb.bin') and os.path.isfile('model/features.bin') and \
               os.path.isfile('model/label.bin') and os.path.isfile('model/target.bin'):
                model = ModelInterface('model/xgb.bin', 'model/label.bin', 'model/features.bin', 'model/target.bin')
                files = [os.path.join(path, file) for file in files]
                print(Fore.GREEN+"Путь существует и есть файлы нужного расширения, процесс запущен")
                r2_score, mse = model.inc_train(files)
                print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            else:
                print(Fore.RED + "Модель не существует, нельзя выполнить дообучение. Сначала выполните обучение")
    else:
        print(Fore.RED+"Введён неправильный путь, введите другой")


#Function that prints a welcome message
def hello():
    mydsa = pyfiglet.figlet_format("MYDSA", font="slant")
    print(to_color(mydsa, "blue"))
    print(Fore.MAGENTA + "MYDSA - Make Your Disk Smart Again\nВерсия утилиты 1000 минус 7")
    print(Fore.YELLOW+"")
