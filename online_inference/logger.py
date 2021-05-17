import logging.handlers

FORMAT = '\n[%(filename)s] - [%(asctime)s] [%(levelname)s]: %(message)s\n'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel("INFO")