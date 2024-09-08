import datetime
import os

import click
import pyfiglet
from colorama import Fore
from rich.console import Console
from rich.table import Table

from src.model import ModelInterface


time_column = 'failure_date'


def to_color(text: str, color: str) -> str:
    """Convert color name to ANSI code and color text

    Args:
        text (str): text to color
        color (str): color name

    Returns:
        str: colored text
    """
    color_code = {
        'blue': '\033[34m',
        'yellow': '\033[33m',
        'green': '\033[32m',
        'red': '\033[31m'
    }
    return color_code[color] + str(text) + '\033[0m'


@click.command()
@click.option("--t_mode",
              prompt="Введите тип процесса - \n1 - predict"
              "(предсказание)\n2 - train(обучение)\n3 - inc_learn(дообучение)"
              "\nq - выход\n->")
def mode(t_mode: str) -> None:
    """Call functions, using their codes, or exit

    Args:
        t_mode (str): can be 1, 2, 3 or q

    Raises:
        SystemExit: if 'q' was entered
    """
    if t_mode == "predict" or t_mode == "1":
        predict()
    elif t_mode == "train" or t_mode == "2":
        train()
    elif t_mode == "inc_learn" or t_mode == "3":
        inc_learn()
    elif t_mode == "q":
        raise SystemExit()
    else:
        print('Не удалось распознать ввод\n')
        mode()


def replace_df(delta: datetime.timedelta) -> str:
    """Get description of failure time in period of 3, 6, 9 or 12 months

    Args:
        delta (datetime.timedelta): timedelta between failure time and date
          of measurement

    Returns:
        str: description
    """
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


@click.command()
@click.option("--path",
              prompt="Введите путь файла, для которого будет"
              " происходит предсказание")
def predict(path: str) -> None:
    """Call prediction function in full-cli mode

    Args:
        path (str): path to csv file with data
    """
    if os.path.isfile(path) and path.endswith(".csv"):
        if os.path.isfile('model/xgb.bin') and os.path.isfile(
            'model/features.bin'
        ) and \
           os.path.isfile('model/label.bin') and os.path.isfile(
               'model/target.bin'
        ):
            model = ModelInterface('model/xgb.bin', 'model/label.bin',
                                   'model/features.bin', 'model/target.bin')
            try:
                df = model.predict(path)
            except ValueError:
                print(Fore.RED + "CSV файл не подходящего формата")
            print(Fore.GREEN +
                  "Путь существует и есть файл нужного расширения, процесс"
                  " запущен")
            size = df.shape
            df = df.sort_values(time_column)
            df['time_delta'] = df[time_column] - datetime.datetime.strptime(
                os.path.basename(path).replace('.csv', ''), '%Y-%m-%d')
            df['time_delta'] = df['time_delta'].apply(replace_df)
            table = Table(title="Disks")
            rows = df.iloc[:15].values.tolist()
            rows = [[str(el) for el in row] for row in rows]
            for column in df.columns:
                table.add_column(column, vertical="middle")
            for row in rows:
                table.add_row(*row, style='bright_green')
            console = Console()
            console.print(table)
            print(
                Fore.YELLOW +
                "Получившиеся значения сохранены в файле result_{}.csv\nВсего"
                " количество элементов - {}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
                    size[0]))
            df.to_csv("result_{}.csv".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")),
                      index=False)
        else:
            print(
                Fore.RED +
                "Модель не существует, нельзя выполнить предсказание. Сначала"
                " выполните обучение")
        print(Fore.YELLOW + "")
        mode()
    else:
        print(Fore.RED + "Файл не того расширения или его не существует")
        print(Fore.YELLOW + "")
        predict()


@click.command()
@click.option(
    "--path",
    prompt="Введите путь директории, для которого будет происходит обучение")
def train(path: str) -> None:
    """Call train function in full-cli mode

    Args:
        path (str): path to directory with csv files
    """
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files) == 0:
            print(Fore.RED +
                  "В директории нет ни одного файла подходящего расширения")
            print(Fore.YELLOW + "")
            train()
        else:
            if not os.path.isdir('model'):
                os.mkdir('model')
            model = ModelInterface()
            files = [os.path.join(path, file) for file in files]
            print(Fore.GREEN +
                  "Путь существует и есть файлы нужного расширения, процесс"
                  " запущен")
            r2_score, mse = model.train(files, 'model/xgb.bin',
                                        'model/label.bin',
                                        'model/features.bin',
                                        'model/target.bin')
            print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            print(Fore.YELLOW + "")
            mode()
    else:
        print(Fore.RED + "Введён неправильный путь, введите другой")
        print(Fore.YELLOW + "")
        train()


@click.command()
@click.option(
    "--path",
    prompt="Введите путь директории, для которого будет происходит дообучение")
def inc_learn(path: str):
    """Call incremental learning function in full-cli mode

    Args:
        path (str): path to directory with csv files
    """
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files) == 0:
            print(Fore.RED +
                  "В директории нет ни одного файла подходящего расширения")
            print(Fore.YELLOW + "")
            inc_learn()
        else:
            if os.path.isfile('model/xgb.bin') and \
               os.path.isfile('model/features.bin') and \
               os.path.isfile('model/label.bin') and \
               os.path.isfile('model/target.bin'):
                model = ModelInterface('model/xgb.bin', 'model/label.bin',
                                       'model/features.bin',
                                       'model/target.bin')
                files = [os.path.join(path, file) for file in files]
                print(
                    Fore.GREEN +
                    "Путь существует и есть файлы нужного расширения, процесс"
                    " запущен")
                r2_score, mse = model.inc_train(files)
                print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            else:
                print(Fore.RED +
                      "Модель не существует, нельзя выполнить дообучение."
                      " Сначала выполните обучение")
            print(Fore.YELLOW + "")
            mode()
    else:
        print(Fore.RED + "Введён неправильный путь, введите другой")
        print(Fore.YELLOW + "")
        inc_learn()


def arg_predict(path: str, sort: bool):
    """Call predict function in simple-cli mode

    Args:
        path (str): path to csv file
        sort (bool): flag to sort result
    """
    if os.path.isfile(path) and path.endswith(".csv"):
        if os.path.isfile('model/xgb.bin') and \
               os.path.isfile('model/features.bin') and \
               os.path.isfile('model/label.bin') and \
               os.path.isfile('model/target.bin'):
            model = ModelInterface('model/xgb.bin', 'model/label.bin',
                                   'model/features.bin', 'model/target.bin')
            try:
                df = model.predict(path)
            except ValueError:
                print(Fore.RED + "CSV файл не подходящего формата")
            size = df.shape
            if sort:
                df = df.sort_values(time_column)
            df['time_delta'] = df[time_column] - datetime.datetime.strptime(
                os.path.basename(path).replace('.csv', ''), '%Y-%m-%d')
            df['time_delta'] = df['time_delta'].apply(replace_df)
            print(
                Fore.YELLOW +
                "Получившиеся значения сохранены в файле result_{}.csv\nВсего"
                " количество элементов - {}".format(
                    datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"),
                    size[0]))
            df.to_csv("result_{}.csv".format(
                datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")),
                      index=False)
    else:
        print(Fore.RED + "Файл не того расширения или его не существует")


def arg_train(path: str):
    """Call train function in simple-cli mode

    Args:
        path (str): path to directory with csv files
    """
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files) == 0:
            print(Fore.RED +
                  "В директории нет ни одного файла подходящего расширения")
        else:
            if not os.path.isdir('model'):
                os.mkdir('model')
            model = ModelInterface()
            files = [os.path.join(path, file) for file in files]
            print(Fore.GREEN +
                  "Путь существует и есть файлы нужного расширения, процесс"
                  " запущен")
            r2_score, mse = model.train(files, 'model/xgb.bin',
                                        'model/label.bin',
                                        'model/features.bin',
                                        'model/target.bin')
            print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
    else:
        print(Fore.RED + "Введён неправильный путь, введите другой")


def arg_inc_learn(path: str):
    """Call incremental learning function in simple-cli mode

    Args:
        path (str): path to directory with csv files
    """
    if os.path.isdir(path):
        files = []
        for File in os.listdir(path):
            if File.endswith(".csv"):
                files.append(File)
        if len(files) == 0:
            print(Fore.RED +
                  "В директории нет ни одного файла подходящего расширения")
        else:
            if os.path.isfile('model/xgb.bin') and \
               os.path.isfile('model/features.bin') and \
               os.path.isfile('model/label.bin') and \
               os.path.isfile('model/target.bin'):
                model = ModelInterface('model/xgb.bin', 'model/label.bin',
                                       'model/features.bin',
                                       'model/target.bin')
                files = [os.path.join(path, file) for file in files]
                print(
                    Fore.GREEN +
                    "Путь существует и есть файлы нужного расширения, процесс"
                    " запущен")
                r2_score, mse = model.inc_train(files)
                print(Fore.GREEN + f"MSE: {mse};   R2: {r2_score}")
            else:
                print(Fore.RED +
                      "Модель не существует, нельзя выполнить дообучение."
                      " Сначала выполните обучение")
    else:
        print(Fore.RED + "Введён неправильный путь, введите другой")


def hello():
    """Show MYDSA welcome message
    """
    mydsa = pyfiglet.figlet_format("MYDSA", font="slant")
    print(to_color(mydsa, "blue"))
    print(Fore.MAGENTA +
          "MYDSA - Make Your Disk Smart Again\nВерсия утилиты 0.5.0")
    print(Fore.YELLOW + "")
