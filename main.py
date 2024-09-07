from src.cli_interface.cli_interface import (hello, mode, arg_inc_learn,
                                             arg_predict, arg_train)
import argparse
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='mydsa', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        '--type',
        choices=['default', 'predict', 'train', 'inc_learn'],
        default='default',
        help='default - запустить интерактивный режим, игнорируются остальные'
        ' аргументы (ПО УМОЛЧАНИЮ)\n'
        'predict - предсказать значения из одного ФАЙЛА формата .csv\n'
        'train - обучение модели на новых данных, находящихся в ДИРЕКТОРИИ\n'
        'inc_learn - дообучение модели на основе данных из ДИРЕКТОРИИ',
        metavar='')
    parser.add_argument(
        '--unsort',
        action="store_false",
        help='Если флаг не установлен, то результат будет сохранён в формате'
        ' от наименьшего срока к наибольшему. Если флаг установлен, то не'
        ' делать сортировку')
    parser.add_argument('--path',
                        type=str,
                        default=os.getcwd(),
                        help='Путь до ФАЙЛА или ДИРЕКТОРИИ')
    args = parser.parse_args()
    if args.type == 'default':
        hello()
        mode()
    else:
        match args.type:
            case 'predict':
                arg_predict(args.path, args.unsort)
            case 'train':
                arg_train(args.path)
            case 'inc_learn':
                arg_inc_learn(args.path)
