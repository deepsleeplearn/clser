from clser.core.run_classification import get_args
from clser.utils.logger import get_logger

logger = get_logger()


if __name__ == "__main__":
    
    file_path = "clser/example/multi_args.yaml"
    
    args = get_args(file_path)
    
    logger.info(
        {"args": args}
    )
    
    
    
    