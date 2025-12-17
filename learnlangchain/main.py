import os
from dotenv import load_dotenv
load_dotenv()

def main():
    print("Hello from learnlangchain!")
    print("OPEN_API_KEY:", os.getenv("OPEN_API_KEY"))


if __name__ == "__main__":
    main()

