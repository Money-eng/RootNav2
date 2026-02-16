import os
import yaml
import copy
import random
from run_training import train

# --- CONFIGURATION ---
NUM_EXPERIMENTS = 50
TEST_ITERS = 750
BASE_CONFIG_FILE = "./training/configs/root_train_cldice.yml"

random.seed(5842) 

search_space = {
    'lr': [5e-4, 1e-4, 5e-5, 1e-5, 5e-6],
    'weight_decay': [1e-3, 1e-4, 1e-5],
    'momentum': [0.8, 0.9],
    'optimizer': ['rmsprop', 'adamw', 'adam']
}

class Args:
    def __init__(self, config_path):
        self.config = config_path
        self.output_example = True
        self.debug = False
        self.resume_iterations = False

def run_random_search():
    if not os.path.exists("./training/configs"):
        os.makedirs("./training/configs")

    with open(BASE_CONFIG_FILE, 'r') as f:
        base_config = yaml.load(f, Loader=yaml.Loader)
    
    if base_config['training']['batch_size'] < 2:
        print("!! ATTENTION: Batch Size forcé à 2 pour éviter plantage BatchNorm !!")
        base_config['training']['batch_size'] = 2

    print(f"--- Démarrage Random Search : {NUM_EXPERIMENTS} essais prévus ---")
    
    previous_choices = set()

    for i in range(1, NUM_EXPERIMENTS + 1):
        opt_name = random.choice(search_space['optimizer'])
        lr = random.choice(search_space['lr'])
        wd = random.choice(search_space['weight_decay'])
        
        while (opt_name, lr, wd) in previous_choices:
            opt_name = random.choice(search_space['optimizer'])
            lr = random.choice(search_space['lr'])
            wd = random.choice(search_space['weight_decay'])
        
        previous_choices.add((opt_name, lr, wd))

        current_config = copy.deepcopy(base_config)
        current_config['training']['optimizer']['name'] = opt_name
        current_config['training']['optimizer']['lr'] = lr
        current_config['training']['optimizer']['weight_decay'] = wd
        
        current_config['training']['train_iters'] = TEST_ITERS
        current_config['training']['val_interval'] = 100
        current_config['training']['print_interval'] = 100

        mom = "NA"
        if opt_name in ['rmsprop', 'sgd']:
            mom = random.choice(search_space['momentum'])
            current_config['training']['optimizer']['momentum'] = mom
        else:
            if 'momentum' in current_config['training']['optimizer']:
                del current_config['training']['optimizer']['momentum']

        print(f"\n=== TEST {i}/{NUM_EXPERIMENTS} ===")
        print(f"Opt: {opt_name} | LR: {lr} | WD: {wd} | Mom: {mom}")

        temp_config_name = f"search_{i}_{opt_name}_lr{lr}_wd{wd}.yml"
        temp_config_path = os.path.join("./training/configs", temp_config_name)
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(current_config, f)
        
        try:
            args = Args(temp_config_path)
            train(args)
            print(f"--> Succès test {i}")
        except Exception as e:
            print(f"!!! ECHEC test {i} : {e}")
            pass

if __name__ == "__main__":
    run_random_search()