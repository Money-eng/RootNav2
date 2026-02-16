import argparse, os
import torch
from models.model_loader import ModelLoader
from run_rootnav import run_rootnav, list_action, info_action
import logging

from collections import OrderedDict

def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

class LogFormatter(logging.Formatter):
    def format(self, record):
        self.datefmt='%H:%M:%S'
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            color = {
                logging.WARNING: 33,
                logging.ERROR: 31,
                logging.FATAL: 31,
                logging.DEBUG: 36
            }.get(record.levelno, 0)
            self._style._fmt = f"[%(asctime)s.%(msecs)03d] \033[{color}m%(levelname)s\033[0m: %(message)s"
        return super().format(record)

logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setFormatter(LogFormatter())
logger.setLevel(logging.INFO)
logger.addHandler(handler)

if __name__ == '__main__':
    # Parser Args
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--list', action=list_action, nargs=0, help='List available models and exit')
    parser.add_argument('-i', '--info', action=info_action, nargs=1, help='Print detail on a single model and exit')
    parser.add_argument('--model', default="arabidopsis_plate", metavar='M', help="The trained model to use (default arabidopsis_plate)")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA')
    parser.add_argument('--segmentation_images', action='store_true', default=True, help='Reduce output files to minimum')
    parser.add_argument('--debug', action='store_true', default=True, help='Show additional debug messages')
    parser.add_argument('input_dir', type=str, default="/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/RootNav2/training/data/test/", help='Input directory', nargs="?")
    parser.add_argument('output_dir', type=str, default="/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/RootNav2/test_output/", help='Output directory', nargs="?")

    args = parser.parse_args()

    if not args.input_dir:
        logger.error("No input folder specified")
        parser.print_help()
        exit()

    if (args.debug):
        logger.setLevel(logging.DEBUG)
        logger.debug("Running in debug mode")
        
    # list files that ends with .pkl in the given model directory
    weights_path = "/home/loai/Documents/code/RSMLExtraction/RSA_reconstruction/Method/RootNav2/inference/models/"
    weights_files = [f for f in os.listdir(weights_path) if f.endswith('.pkl')]
    weights_files = sorted(weights_files, key=lambda x: os.path.getmtime(os.path.join(weights_path, x)), reverse=True)
    
    
    try:
        model_data = ModelLoader.get_model(args.model)
    except Exception as ex:
        logger.error(ex)
        exit()
    
    
    for f in weights_files:
        logger.info(f"Processing weights: {f}")
        
        model_data['configuration']['network']['weights'] = f
        
        full_weight_path = os.path.join(weights_path, f)
        checkpoint = torch.load(full_weight_path, map_location='cuda', weights_only=False)
        
        if 'model_state' in checkpoint:
            raw_state = checkpoint['model_state']
        else:
            print('Different format')
            raw_state = checkpoint
            
        state = convert_state_dict(raw_state)
        model_data['model'].load_state_dict(state)
        model_data['model'].eval()
        
        output_dir = ''
        if not args.output_dir:
            logger.info("No output folder specified, will try and write output to " + args.input_dir + "_output")
            output_dir = args.input_dir + '_output'
        else:
            output_dir = args.output_dir + "/" + f.split('.')[0]

        if os.path.exists(output_dir):
            if os.listdir(output_dir):
                logger.warning(f"{output_dir} already exits and isn't empty, files may be overwritten")
            else:
                logger.debug(f"{output_dir} already exits and is empty")
        else:
            logger.debug(f"Creating output directory {output_dir}")
            os.makedirs(output_dir)
            
        # Process (Le modèle a maintenant les nouveaux poids)
        run_rootnav(model_data, True, args, args.input_dir, output_dir)