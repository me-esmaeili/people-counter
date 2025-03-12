from PeopleCounter import *
def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='People Counter')
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration file')
    args = parser.parse_args()

    # Create and run the people counter
    counter = PeopleCounter(args.config)
    counter.run("assets/samples/20231207153936_839_2.avi")


if __name__ == "__main__":
    main()