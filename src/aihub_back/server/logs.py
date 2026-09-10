import logging


# ----------------------------------Logger----------------------------------
def get_logger():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] : %(message)s",
        filename="logs.log",
        filemode="w",
    )
    return logging.getLogger(__name__)