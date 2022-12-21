from visualize import *
import sys
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default=None, help="the input json files, split with ','")
    parser.add_argument('--jsonl', type=str, default=None, help="the input jsonl files, each line is an input json file")

    args = parser.parse_args()
    if args.jsonl:
        eval_jsonl = args.jsonl
        multi_input = False
    elif args.json:
        eval_jsonl = args.json
        multi_input = True
    else:
        print('Invalid arguments: empty input files.\n')
        return
    eval_board = EvaluationBoard(eval_jsonl=eval_jsonl, multi_input=multi_input)
    eval_board.visualize()

if __name__ == "__main__":
    main()