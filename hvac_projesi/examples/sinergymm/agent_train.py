import yaml
import sys

import numpy as np

from ReinforcementLearning.QLearningAgent import QLearningAgent
from ReinforcementLearning.environment import ObservationWrapper, create_environment


def main(config_path):
    #config ayarlarımızı atıyoruz
    with open(config_path, "r") as conf_yml:
        config = yaml.safe_load(conf_yml)

    #ajanı eğitirken ona vereceğimiz ve filtreleyeceğimiz verileri atıyoruz
    obs_to_keep = np.array(config["env"]["obs_to_keep"])
    mask = np.array(config["env"]["mask"])

    #environment'ımızı oluşturuyoruz
    env = create_environment(config["env"])
    env = ObservationWrapper(env, obs_to_keep)

    #ajanımızı oluşturuyoruz
    agent = QLearningAgent(env, config["agent"], mask)

    #ajanızımı eğitiyoruz
    agent.train()
    #sonuçlarımızı kaydediyoruz
    agent.save_results()
    #açtığımız environment'i kapatıyoruz
    env.close()

if __name__ == "__main__":
    main(sys.argv[1])