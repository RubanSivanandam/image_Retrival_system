import argparse, json, pprint
from src.crew import RetrievalCrew
from src.config.settings import *

def main():
    p = argparse.ArgumentParser(description="CrewAI Garment Search")
    p.add_argument("--index", action="store_true", help="build index")
    p.add_argument("query", nargs="*", help="search phrase")
    args = p.parse_args()

    crew = RetrievalCrew()

    if args.index:
        crew.build_index(); return

    if not args.query:
        print("Give a query or --index"); return

    res = crew.search(" ".join(args.query))
    try:
        data = json.loads(res)
        pprint.pp(data["results"])
    except Exception:
        print(res)

if __name__ == "__main__":
    main()
