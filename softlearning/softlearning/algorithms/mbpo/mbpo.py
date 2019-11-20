import gym

def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--env-name', default="Hopper-v2",
        help='Mujoco Gym environment (default: Hopper-v2)')


def main():
    readParser()


if __name__ == '__main__':
    main()
