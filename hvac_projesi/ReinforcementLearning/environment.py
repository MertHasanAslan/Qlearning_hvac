"""
Bu dosya agent'imizi çalıştıracağımız environment'i oluşturmamızı sağlayacak.
gym ve sinergym import ederek oluşturacağız.
"""
import gym #environment için gerekiyor
import numpy as np

#sineygym'i hvac için gereken environment'i kurmak için kullanacağız.
from sinergym.utils.wrappers import NormalizeObservation, MultiObsWrapper
from sinergym.utils.constants import (
    RANGES_5ZONE,
    RANGES_DATACENTER,
    RANGES_WAREHOUSE,
    RANGES_OFFICE,
    RANGES_OFFICEGRID,
    RANGES_SHOP
)


class ObservationWrapper(gym.ObservationWrapper): #gym.ObservationWrapper class'ından miras alan bir class oluşturuyoruz.
    def __init__(self, environment, obs_to_keep):
        """
        environment: agent'imizi eğiteceğimiz bölge. (sinergym_config.yaml dosyasında env kısmı olarak da görebiliriz)
        obs_to_keep: agent'imizin eğitiminde kullanılacak gözlem değişkenlerinin endekslerini içerir. (yani sıcaklık ve nem gibi değişkenler) (offset by 4)
        """
        super().__init__(environment)
        self.environment=environment

        #Zamanla ilgili endeksleri gözlemlememek için aşağıdaki fonksiyonu yazıyoruz. (offset by 4)
        if obs_to_keep.any() == False: #eğer ki liste boşsa zamanla ilgili değişkenleri dışarıda bırakmak için 4'den başlayıp son veriye kadar gider.
            self.obs_to_keep = np.arange(4, environment.observation_space.shape(0))
        else: #eğer ki gözlemlenecek değişkenlerin endeksleri listeye girilmişse her biri 4 sağa kaydırılır ki zamanla ilgili değişkenler dışarıda kalsın.
            self.obs_to_keep = np.add(np.array(obs_to_keep), 4)


    def observation(self, observation): #Bu fonksiyon obs_to_keep ile belirlediğimiz değişkenleri döndürür.
        #observation: tüm observationlar.
        return observation[self.obs_to_keep] #bu sayede observationlar içerisinden endekslerini belirlediğimiz observationları döndüreceğiz.


def create_environment(environment_config: dict = None) -> gym.Env: #Bu fonksiyon gym.Env türünde return yapar.
    #environmennt_config parametresi ile sinergym_config.yaml kullanılarak env: bölümünden name:'i çekeceğiz ve gym class'ından environment oluşturacağız.

    if environment_config == False:
        environment_config = if_config_is_null() #fonksiyon sayesinde oluşturacağımız env için isim atadık

    environment = gym.make(environment_config["name"]) #Environment oluşturuldu.

    ranges = environment_range_find(environment_config) #izleme aralığını bulacağız

    environment = observation_maker(environment_config, ranges, environment) #sonunda environment'imiz istediğimiz aralıklarda izlenecek şekilde oluştu

    return environment


def if_config_is_null(): #eğer ki environment config boşsa environment oluşturmak için bir isim atayacağız.
    environment_config = {"name": "Eplus-5Zone-hot-discrete-v1"} #Bu sayede sonrasında gym class'ını kullanarak env oluşturacağımız ismi ayarlamış olduk.
    return environment_config


def environment_range_find(environment_config): #Environment'imizi izlerken hangi aralıklarda izleyeceğimizi config'deki isme göre kararlaştıracağız.
    #environment_config parametresi bkz:sinergym_config.yaml

    #oluşturacağımız environment'in izleme aralığını seçiyoruz.

    if "normalize" in environment_config and environment_config["normalize"] is True:
        environment_name = environment_config["name"].split("-")[1] #ismi ayırıyoruz ve ismine göre aralığını seçiyoruz.
        if environment_name == "datacenter":
            ranges = RANGES_DATACENTER
        elif environment_name == "5Zone":
            ranges = RANGES_5ZONE
        elif environment_name == "warehouse":
            ranges = RANGES_WAREHOUSE
        elif environment_name == "office":
            ranges = RANGES_OFFICE
        elif environment_name == "officegrid":
            ranges = RANGES_OFFICEGRID
        elif environment_name == "shop":
            ranges = RANGES_SHOP
        else:
            raise NameError(f"env_type {environment_name} is not valid, check environment name") #böyle bir isim bulunmyuyorsa bu hatayı alacağız.
        return ranges #izleme aralığını döndürüyoruz.

def observation_maker(environment_config, ranges, environment): #Bu fonksiyon sayesinde environmentimiz için observation oluşturacağız.
    #gözlem türümüzü seçip ona göre birini oluşturacağız.

    if "normalize" in environment_config and environment_config["normalize"] is True:
        environment = NormalizeObservation(environment, ranges = ranges)

    if "multi_observation" in environment_config and environment_config["multi_observation"] is True:
        environment = MultiObsWrapper(environment)

    return environment
