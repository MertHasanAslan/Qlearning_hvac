"""
Öğrenme işleminde kullanmak için bir Q learning Agent oluşturacağız
"""
import os
import logging
from datetime import datetime, date
import numpy as np
from ReinforcementLearning.hnp import HNP

from sinergym.envs import EplusEnv

logging.getLogger().addHandler(logging.StreamHandler()) #default gelen logger'a StreamHandler ekliyoruz. Artık log mesajları ekrana yazdırılabilinir.
logger = logging.getLogger() #logger nesnesini oluşturduk
logger.setLevel("INFO") #Farklı log seviyeleri vardır ve önem sıralarına göre ayrılırlar. Artık Info ve üstü derecede önemli mesajlar kullanılacak

#bu kodda obs ile state aynı anlama geliyor diyebiliriz

class QLearningAgent:
    """
    Reinforcement Learning için kullanacağımız Q-Learning Agent'ini bu class ile oluşturacağız
    """
    #environment EplusEnv türünde çünkü sinergym'den türetilmiş bir environment kullanacağız.(EnergyPlus)
    def __init__( self, env: EplusEnv, config: dict, obs_mask: np.ndarray, results_dir: str = "training_results", beobench_check: bool = False, hnp_check: bool = True) -> None:

        self.env = env #Bu Agentimizi çalıştıracağımız environment olacak
        self.config = config #config ayarlarımız. bkz: examples/sinergymm/sinergym_config.yaml
        self.results_dir = results_dir #Ödüllerimizi save_results fonksiyonuyla kaydederken kaydedeceğimiz yer. (default = training_results)
        self.rewards = [] #Ödüllerimizi kaydedeceğiz
        self.beobench_check = beobench_check #beobench bool'unun True olup olmadığı
        self.hnp_check = hnp_check #HNP bool'unun true olup olmadığı
        self.gamma = config["gamma"] #gamma oranı ile gelecekteki ödüllerin önemini belirleyeceğiz
        self.epsilon = config["initial_epsilon"] #Ajanın hangi olasılıkla rastgele eylem yapacağını belirler. 1 ile başlar ve yavaşça düşer
        self.learning_rate = config["learning_rate"] #Ajanın hangi olasılıkla öğreneceğini belirler
        self.epsilon_annealing = config["epsilon_annealing"] #Ajanın rastfele eylem yapma olasılığının düşma katsayısı.
        self.learning_rate_annealing = config["learning_rate_annealing"]  #zamanlar öğrenme oranını azaltacak olan katsayı. İlerdedikçe daha büyük yerine küçük adımlar atacak.
        self.obs_mask = obs_mask #gözlenebilir değişkenleri yavaş, hızlı ve ayrık olarak üçe ayırmak için kullanılacak

        self.continuous_index, self.discrete_index = self.mask_filter(self.obs_mask)
        self.permutation_index = np.hstack((self.continuous_index, self.discrete_index)) #sürekli ve ayrık endeksleri yanyana birleştirdik
        self.num_tiles = self.make_num_tiles(config)
        self.continuous_low, self.continuous_high = self.make_continuous_bounds(self.continuous_index)

        self.obs_space_shape = self.get_obs_shape() #sürekli ve ayrık değişen değişkenlerin şekillerinin birleşimi.
        self.act_space_shape = self.get_act_shape() #ajanın yapabileceği aksiyon boyutunu döndürür. (ajanın seçebileceği fakrlı eylem sayısı)
        
        #saklanan q değerlerine göre ödül seçilir.
        self.q_table = np.zeros((*self.obs_space_shape, self.act_space_shape)) #q tablosu oluşturuyoruz. gözlem x aksiyon büyüklüğünde #durum eylem comb. için ödülleri saklar
        
        self.value_table = np.zeros(self.obs_space_shape) #durumların max değerini saklar. q table'daki durum eylem çiftlerine göre en yüksek ödül seçilir ve saklanır.
        self.state_visitation = np.zeros(self.value_table.shape) #hangi durumları kaç kere ziyaret ettiğini saklıyoruz
        self.average_rewards = [] #ortalama ödülleri saklar

        if self.hnp_check:
            self.hnp = HNP(np.where(obs_mask == 0)[0])


    def get_next_value(self, obs: np.ndarray) -> tuple[float, np.ndarray]:
        """

        :param obs: mevcut hvac sistemindeki durumu temsil eden bir gözlem dizisi
        :return: next_value: bu gözlem için hesaplanan sonraki değer. (en iyi eylemin ödülü)
        :return: full_obs_index: bu gözleme karşılık gelen value_table indexi. (value_table içindeki yeri)
        """

        #value_table içinde kullanılacak olan tamsayı indexini döndürür. (sürekli ve ayrık değişkenleri içeren)
        #sürekli değişen değişkenlerin float (hassas olan) indexlerini döndürür. HNP işlemlerinde kullanılabilinir.
        full_obs_index, continuous_obs_index_floats = self.get_value_table_index_from_obs(obs)

        # mevcut durumun (indeksinin) value_table içinde karşılık gelen değerini (en yüksek değer) alır. bu değer mevcut durum için en yüksek değerdir.
        next_value = self.value_table(full_obs_index)

        if self.hnp_check: #HNP kullanacaksak
            #hnp sınıfında tanımladığımız sonraki değeri hesaplamak için oluşturduğumuz fonksiyonu çağırıyoruz.
            next_value = self.hnp.get_next_value(self.value_table, full_obs_index, continuous_obs_index_floats)

        # mevcut durum için valut_table'da karşılık gelen en yüksek değeri veya hnp hesaplaması ile aldığımız değeri ve mevcut durumun indeksini döndürüyoruz
        return next_value, full_obs_index


    def get_value_table_index_from_obs(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """

        :param obs: mevcut durumu temsil eden bir gözlem dizisi
        :return: bir observation'a göre value_table içerisinden değer döndürür
        """

        obs = obs[self.permutation_index] #self.permutation_index kullanılarak observation dizisi sıralanır
        continuous_obs = obs[: len(self.continuous_index)] #bu dizi içerisinden sadece sürekli değişen değişkenler tutulur

        continuous_low = continuous_obs - self.continuous_low #sürekli değişen değişkenlerin alt sınırı
        continuous_interval = self.continuous_high - self.continuous_low #sürekli değişen değişkenlerin aralığı

        # observation interval. sürekli değişen değişken sayısı kadar value_Table'dan boyut alır
        # diyelim ki nem ve sıcaklık ve aralıkları da 6 ve 11 olsun. sıfır tabanlı yapabilmek için de -1 dedik
        #artık bu aralığımız 0-5 ve 0-10 :)
        obs_interval = (np.array(self.value_table.shape[: len(self.continuous_high)]) - 1) #contimnuous_high dememizin sebebi sürekli değişen değişken sayısı kadar olması low da olurdu

        continuous_obs_index_floats = ((continuous_low) / (continuous_interval) * (obs_interval)) #sürekli değişen değişkenlerin indexlerini float şeklinde hesaplar
        continuous_obs_index = np.round(continuous_obs_index_floats) #bulduğumuz floatları integer'a yuvarlıyoruz

        #sürekli değişen değişkenler ile ayrık değişen değişkenleri yatay olarak birleştirdik.
        #sistemin şu anki durumunu temsiz eden bir tam sayı dizisi oluşturduk
        #bu full?obs_index'i daha sonrasında q_table/value_table içinde kullanarak ilgili değere ulaşabileceğiz.
        full_obs_index = np.hstack((continuous_obs_index, obs[len(self.continuous_index) :])).astype(int)

        return full_obs_index, continuous_obs_index_floats


    def get_act_shape(self) -> int:
        #ajanın yapabileceği aksiyon boyutunu döndürür. (ajanın seçebileceği fakrlı eylem sayısı)
        return self.env.action_space.n


    def make_tile_coded_space(self, tile_size):

        tile_coded_space = [] #tile (fayanslama) işlemi yapılmış olan sürekli değişkenleri saklayacağız

        for i in self.continuous_index: #sürekli değişen değişkenlerin indexlerinde dolaşacağız

            current_tile_size =  tile_size[i] #şu anki sürekli değişen değişkenin bir fayans uzunluğu

            tiles = np.arange(0, 1 + current_tile_size, current_tile_size) #0 ve 1 arasında fayanslama işlemi gerçekleşiyor. np.arange ile son sınır dahil edilmediğinden +current_tile_size

            tile_coded_space.append(tiles) #fayanslanmış diziyi ekliyoruz.

        return tile_coded_space


    def get_obs_shape(self) -> tuple:
        """


        :return: combined_shapes = sürekli ve ayrık değişen değişkenlerin şekillerinin birleşimi.
        """

        #tile_size sürekli değişkenlerin sayısı kadar uzun bir array olacak (çünkü num_tiles bu uzunlukta bir array).
        tile_size = 1/self.num_tiles #tiles (fayans) sayısı arttıkça boyutu küçülür (Tiles sürekli değişkenlerin alanını 0 ve 1 arasında ayrılan parçalardır)

        tiles_coded_space = self.make_tile_coded_space(tile_size)

        #sürekli değişen değişkenlerin aralık uzunluklarını ekledik
        continuous_shapes = []
        for tiles in tiles_coded_space:
            continuous_shapes.append(len(tiles))

        #ayrık değişen değişkenlerin maksimum değerlerini aldık
        discrete_shapes = []
        for index in self.discrete_index:
            discrete_shapes.append(self.env.observation_space.high[index])

        #sürekli ve ayrık değişen değişkenlerin şekillerini birleştirdik
        combined_shapes = continuous_shapes + discrete_shapes

        return tuple(combined_shapes)


    def make_continuous_bounds(self, continuous_index):
        """
        Normilizasyon işlemi için alt ve üst sınırı 0 ve 1 olarak seçeceğiz. Bu sayede işlem yaparken daha rahat olacak.
        Genel olarak bu tarz projelerde 0 ve 1 aralığı tercih ediliyor.
        """
        continuous_low = np.zeros(len(self.continuous_index))
        continuous_high = np.ones(len(self.continuous_index))

        return continuous_low, continuous_high


    def mask_filter(self, obs_mask):
        """

        :param obs_mask: gözlenebilirlerin listesi. filtreleme için kullanacağız. (0 = yavaş sürekli, 1= hızlı sürekli, 2 = ayrık)
        :return: continuous_idx (yavaş ve hızlı sürekli değişen değişkenler), discrete_idx (ayrık değişen değişkenler)
        """
        continuous_index = np.where(obs_mask <= 1)[0] #burada 0 ve 1'ler yavaş ve hızlı sürekli değişen değişkenler olduğu için o kısmı alıyoruz.
        discrete_index = np.where(obs_mask == 2)[0] #burada 2'ler ayrık değişen değişkenler olduğu için bu kısmı alıyoruz.

        return continuous_index, discrete_index

    def make_num_tiles(self, config): #(Tiles sürekli değişkenlerin alanını 0 ve 1 arasında ayrılan parçalardır)
        """

        :param config: bkz sinergym_config
        :return: Tiling işlemi için kaç parçaya ayıracağımızı döndüreceğiz (numpy array'i şeklinde) sürekli değişkenleri 0 ve 1 arasında parçalara ayırıyoruz
        """

        if type(config["num_tiles"]) is list: #eğer ki listeyse iş kolay. direkt array haline getirip döndürebiliriz ama bizim config ayarlarımızda direkt bir integer
            num_tiles = np.array(config["num_tiles"])
        else:
            #eğer bir liste değilse veya boşsa
            num_tiles = np.full(self.continuous_index.shape, config["num_tiles"]) #sürekli değişkenlerin boyutunda bir numpy array'i oluşturur.

        return num_tiles


    def choose_action(self, obs_index: np.ndarray, mode: str = "explore") -> int:
        """

        :param obs_index: observation space'de mevcut durum
        :param mode: fonksiyonun nasıl çalışacağını belirtir. explore ve ya greedy olarak seçilir.
        :return: q tablosunda seçilen eylemi döndürür
        """

        if mode == "explore": #Ajanımız keşif yanı rastgele eylem yapma modunda ise
            if np.random.rand(1) < self.epsilon: # epsilon değeri ajanımızın rastgele bir eylem yapıp yapmayacağını belirler. bkz: sinergym_config.yaml
                return self.env.action_space.sample() #rastgele bir eylem döndürülür.
            #eğer ki rastgele bir eylem seçmediyse
            max_action = np.argmax(self.q_table[tuple(obs_index)]) #bu durumdaki maksimum değerdeki eylemi seçer.
            return max_action #q table üzerindeki anlık durumumuzdaki en yüksek değerli eylemi döndürür. (greedy modunda gibi max değeri seçerek hamle yapar)

        elif mode == "greedy":
            max_action = np.argmax(self.q_table[tuple(obs_index)])  # bu durumdaki maksimum değerdeki eylemi seçer.
            return max_action  # q table üzerindeki anlık durumumuzdaki en yüksek değerli eylemi döndürür.


    def save_results(self) -> None:
        """
        Bu Fonksiyon date bilgilerine göre ödülleri kaydetme işlemi yapıyor. Bu sayede daha sonra istediğimiz bir zaman kullanabileceğiz.
        """

        today = date.today()
        day = today.strftime("%Y_%b_%d")
        now = datetime.now()
        time = now.strftime("%H_%M_%S")
        base_dir = "root" if self.use_beobench else os.getcwd()
        dir_name = f"/{base_dir}/{self.results_dir}/{day}/results_{time}"
        os.makedirs(dir_name)

        logging.info("Saving results...")

        with_hnp = ""
        if self.hnp_check:
            with_hnp = "-HNP"

        np.savez(
            f"{dir_name}/{self.__class__.__name__}{with_hnp}_results.npz",
             qtb=self.q_table,
             rewards=self.rewards,
        )


    def train(self):
        """
        Q-learning algoritması kullanarak ajanımızı eğiteceğiz

        :return:None yani değer döndürmeyeceğiz. ajanımızı eğitecek olan bir fonksiyon bu.
        """

        obs = self.env.reset() #ortamı sıfırlıyoruz ve başlangıç observation'unu alıyoruz.
        old_value_table_index, prev_continuous_obs_index_floats = self.get_value_table_index_from_obs(obs)
        episode_reward = 0 #her bölümde kazanılan reward'ı saklar.
        episode_num = 0   #kaçıncı bölümde olduğumuz
        steps_num = 0     #Bölümün içindek açıncı adımda olduğumuz

        #config ayarlarında kaç bölüm boyunca ajanımızı eğiteceğimizi yazmıştıkç
        while episode_num < self.config["num_episodes"]: #while döngüsü bölüm sayısı kadar çalışacak.

            # ajan value table index'ine göre bir aksiyon belirleyecek
            # öğrenme modunda olduğumuz için de explore modunda çalıştıracağız.
            action = self.choose_action(old_value_table_index, mode = "explore")

            #bu indexteki en yüksek ödülü q table'dan alıyoruz ve value table içinde saklıyoruz.
            self.value_table = np.nanmax(self.q_table, axis = -1)

            #aksiyonumuzu self.env.step fonksiyonunun içine atıyoruz
            #bize yeni bir durum (obs yeni sıcaklık fan durumu vs.) ödül (reward) bitip bitmediğine dair bir bool (end) ve ekbilgiler döndürür (additional_info)
            obs, reward, end, additional_info = self.env.step(action)
            episode_reward = reward + episode_reward #bu bölümde kazanılan toplam ödüle bu ödülü de ekliyoruz.

            #elde ettiğimiz yeni observation değerini get_next_value fonksiyonuna atıyoruz
            #value_table içindeki değeri ve o değerin value_table içindeki indeksini döndürüyoruz.
            #new_value eşittir şu anki durumda yapılacak en iyi eylemin ödülü ve new_value_table_index bu state'deki value_table içindeki indeksi temsil eder
            new_value, new_value_table_index = self.get_next_value(obs)


            #önceki durum ve eylem (action) çifti ile önceki q tablosundaki indeksini belirliyoruz
            old_q_table_index = tuple([*old_value_table_index, action])

            #bu durumun ziyaret sayısını 1 arttırıyoruz
            self.state_visitation[old_q_table_index[: -1]] += 0 #old_q_table_index[:-1] aksiyonu hariç tutmak için [: -1] yapıyoruz
            current_q = self.q_table[old_q_table_index] #şu anki indeksteki q tablosunda karşılık gelen değeri alıyoruz

            #hedef q değerini hesaplıyoruz
            #gamma = discount (indirim faktörü) bu değer 0-1 arasında bulunur ve ajanın anlık mı gelecekteki ödüle mi daha çok önem vereceğine karar verir
            #gamma 1'e yakınsa ajan gelecekteki ödüllere 0'a yakınsa şimdike ödüle daha çok önem verir
            #new_value sonraki durumda (mevcut eylemi gerçekleştirince) alacağı max ödülü temsil eder.
            #q_target = mevcut ödül ve sonraki aksiyon ile alınabilinecek max ödül * gamma'dır. bu bir q learning formülüdür.
            #bu değer q tablosunda güncellenecek olan değerdir.
            q_target = self.gamma * new_value + reward

            #q tablosunda bu durumun karşılık geldiği yere yeni değer yazılır
            #formül = Q(s,a) = Q(s,a) + learning_Rate * (q_target - Q(s,a))
            #bu sayede q tablosundaki bu indeksteki değer güncellenir.
            #mevcut q değeri hedeflenen q değerine daha yakın olacak şekilde ayarlanır. bu da doğru hamleyi seçmemizde yardımcı olacaktır.
            self.q_table[old_q_table_index] = current_q + self.learning_rate * (q_target - current_q)
            steps_num = steps_num + 1 #bölüm içindeki adım sayısını bir arttırdık.
            old_value_table_index = new_value_table_index #yeni durumumuzun value table indeksini güncelledik.

            if steps_num == self.config["horizon"]: #eğer ki bölüm içindeki adım sayısı (bir gündeki saat sayısı) sonlandıysa (gün bittiyse)

                average_reward = (episode_reward / steps_num)
                #ödüller gitgide artarsa ajan doğru öğreniyor demektir.
                self.average_rewards.append(average_reward)
                self.rewards.append(episode_reward) #ödüller listesine bu bölümün ödülün kaydediyoruz
                if episode_num % 10 == 0: #her 10 bölümde (10 gün'de) bir ekrana bilgiler yazdırılır
                    logger.info("Episode %d --- Reward: %d Average reward per timestep: %.2f",episode_num, episode_reward, average_reward)


                #yeni bir bölüme geçmeden önce sıfırlama işlemlerini yapıyoruz ve episode sayısını bir arttırıyoruz
                episode_num = 1 + episode_num
                episode_reward = 0
                steps_num = 0

                self.learning_rate = self.learning_rate_annealing * self.learning_rate
                self.epsilon = self.epsilon_annealing * self.epsilon #ajanın rastgele aksiyon seçme olasılığını düşürüyoruz. böylece son episode (gün)'lere doğru daha fazla greedy' yapar

            if end: #eğer ki bölüm bittiyse
                obs = self.env.reset()  # ortamı sıfırlıyoruz ve başlangıç observation'unu alıyoruz.
                old_value_table_index, prev_continuous_obs_index_floats = self.get_value_table_index_from_obs(obs)
