from identifier import Identifier
import argparse
import os

from config import *
argparser = argparse.ArgumentParser()

argparser.add_argument('-type', "--type", help="clasic | jigsaw \t specify the type of the tests")
argparser.add_argument('-bonus', "--bonus", action='store_true', help='add if you want to test with bonus' )
argparser.add_argument('-nr', '--test_number', required=False, type=int, help='number of tests')

args = argparser.parse_args()

if __name__ == '__main__':
    os.chdir('.')
    type = args.type
    bonus = args.bonus
    nr = args.test_number
    total_nr_tests = len(list(filter(lambda name: name.endswith('.jpg'), [name for name in os.listdir(os.path.join(PATH_TESTS, type))])))
    if nr is None:
        nr = total_nr_tests
    else:
        nr = min(nr, total_nr_tests)
    print('Bonus =',bonus)
    identifier = Identifier()
    for i in range(1, nr + 1):
        answer = identifier.process_image(i, type=type, bonus=bonus)
        print('Test', i)
        print(answer)
        if bonus:
            filename = str(i) + SUFIX_BONUS
        else:
            filename = str(i) + SUFIX_NO_BONUS
        f = open(os.path.join(os.path.abspath(PATH_SOLUTIONS), OWNER_NAME, type, filename), 'w')
        f.write(answer)
        f.close()
        print('Answer ' + str(i) + ' written')

        