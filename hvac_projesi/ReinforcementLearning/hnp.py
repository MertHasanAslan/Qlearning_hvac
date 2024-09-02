"""

HNP class'ını oluşturacağız (slow_continuous_index'ler için)
HNP Hesaplamaları yavaş değişen sürekli değişkenlerde kullanılır bu yüzden HVAC projesinde kullanılacaktır.

"""

import numpy as np


class HNP: #Hyperspace Neighbour Penetration
    """
    Bu class hnp işlemleri için oluşturulacak ve bu işlemler daha sonrasında Q-Learning methodu ile en yüksek değerli hamleyi bulmak için kullanılacak.
    """
    def __init__(self, slow_continuous_index):
        """
        init fonksiyonumuz boş döndürür ve hnp sınıfımız için bir inşaa görevi görür (constructor)
        slow_continuous_index parametresi yavaş değişen değişkenlerin indeksleri
        """
        self.slow_continuous_index=slow_continuous_index

        n_slow_continuous_index = len(slow_continuous_index) #Bu sayede slow_continuous_index'in içinde kaç tane değişken olduğunu bulacağız.

        if n_slow_continuous_index > 0:
            portion_index_matrix = self.make_portion_index_matrix(n_slow_continuous_index)
            self.all_portion_index_combos = self.make_all_portion_index_combos(n_slow_continuous_index, portion_index_matrix)


    def get_next_value(self, value_table, full_obs_index, cont_obs_index_floats):
        """
        HNP hesaplamaları yalnızca yavaş değişen ve aralıklarda yer alabilen (float) değişkenler için kullanılır
        örneğin gün içindeki bir mekanın sıcaklık/nem gibi değişkenleri yavaş yavaaş değişeceği ve float değerlerde
        olabileceği için HNP hesaplamaları için uygundur. Bu da hvac projesinde HNP hesaplamalarını kullanabileceğimiz anlamına geliyor.


        :param value_table: Durum değer tablosu. Durumlara karşılık gelen değerleri içerir
        :param full_obs_index: Gözlemin durum değer tablosundaki indeksidi
        :param cont_obs_index_floats: Sürekli değişen değişkenleri indeksleri
        :return: next_value: Sonraki değeri döndürüyoruz (sürekli değişen değişkenler için)
        """

        if len(self.slow_continuous_index) == 0: #Eğer ki yavaş değişen değişken yoksa HNP hesaplaması yapılmayacak
            return value_table[tuple(full_obs_index)] #şu anki durum değerini döndürüyoruz

        slow_continuous_obs_index_floats = self.make_slow_continuous_obs_index_floats(self.slow_continuous_index, cont_obs_index_floats)
        slow_continuous_obs_index_int_below = self.make_slow_continuous_obs_index_int_below(slow_continuous_obs_index_floats)
        slow_continuous_obs_index_int_above = self.make_slow_continuous_obs_index_int_above(slow_continuous_obs_index_floats)
        all_value_table_index_combos = self.make_all_value_table_index_combos(slow_continuous_obs_index_int_below, slow_continuous_obs_index_int_above)
        portion_below, portion_above, portion_matrix = self.portion_calculator(slow_continuous_obs_index_int_above, slow_continuous_obs_index_floats)

        non_hnp_index = full_obs_index[len(self.slow_continuous_index):] #full_obs_index içindeki HNP hesaplamalarının yapılmayacağı yerler (discrete, fast vs.)

        next_value = 0
        for index, combo_value in enumerate(self.all_portion_index_combos): #enumerate ile all_portion_index_combos içerisindeki index ve value'lar döndürülecek
            #combo_value all_portion_index_combos içerisinde bulunan 0 ve 1 lerden oluşan bir değerdir örnek [0, 0] ve ya [1, 0]...

            portions = portion_matrix[np.arange(len(slow_continuous_obs_index_floats)), combo_value]
            #np.arange(len(slow_continuous_obs_index_floats)) ile 0 dan hnp işlemi yapılacak indekslerin uzunluğu kadar bir array oluşturuyoruz
            #oluşan array satırları combo_value'lar ise sutunları seçer
            #portions bu durumda sırasıyla her bir satırdan combo_value'deki değere göre (0 ya da 1) bir değer seçilecek. buna göre ağırlıklara eşit olacak.

            value_from_value_table = value_table[tuple(np.hstack((all_value_table_index_combos[index], non_hnp_index)).astype(int))]
            #alt ve üst değerler ile bulduğumuz tüm kombinasyonlardan oluşan all_value_table_index_combos'un index'deki satırını non_hnp_index ile yatay olarak birleştirir tamsayı array oluşturur
            #bu oluşan array value_table tablosu (durum değer tablosu) içindeki bir değeri işaret eder ve value_from_value_table bu değere eşit olur

            next_value += np.prod(portions) * value_from_value_table
            #np.prod(portions) ile portions içindeki ağırlıkların çarpımını buluruz (combo_value içindeki 0 ve 1 ler ile ağırlıkları seçmiştik)
            #Durum değer tablosundan döndürdüğümüz value_from_value_table'değeri ile bu ağırlık çarpılır ve next_value'ya eklenir

        return next_value #for döngüsü bittikten sonra değerlerin eklenmesiyle oluşan sonraki değerimizi döndürüyoruz


    def make_portion_index_matrix(self, n_slow_continuous_index): #bu fonksiyonu daha sonrasında olası tüm kombinasyonları bulmak için kullanacağız.
        """
        :param n_slow_continuous_index:
        :return: portion_index_matrix (yavaş değişen değişken sayısı uzunluğunda 0 ve 1 lerden oluşan bir dataframe döndürüyor)
        """
        #np.vstack kullanarak üst üste dizdik örnek = [[0, 0], [1, 1]]
        #sonrasında .T ile transpoze ettik yani satırlar ve sutunlar yer değiştirdi örnek = [[0, 1], [0, 1]]
        #return edildi
        portion_index_matrix = np.vstack((np.zeros(n_slow_continuous_index), np.ones(n_slow_continuous_index))).T

        print("Portion index matrix: {}".format(portion_index_matrix))
        return portion_index_matrix #örnek output = [[0. 1.], [0. 1.]] (2 değişken varsa)


    def make_all_portion_index_combos(self, n_slow_continuous_index, portion_index_matrix): #Oluşabilecek tüm kombinasyonların dataframe'ini oluşturacağız
        """

        :param n_slow_continuous_index: #yavaş değişen değişkenlerin sayısı
        :param portion_index_matrix: #yavaş değişen değişken sayısı uzunluğunda 0 ve 1 lerden oluşan bir dataframe (bununla tüm kombinasyonları bulacağız)
        :return: all_portion_index_combos (0 ve 1 leri kullanarak oluşabilecek tüm kombinasyonları göreceğiz.)
        """
        #*portion_index_matrix  (yıldızı kullanarak) unpacking işlemi yapıyoruz. bu sayede her bir sütun ayrı ayrı np.meshgrid fonksiyonundan geçecek.
        #aslında şu anlama geliyor np.meshgrid(portion_index_matrix[:, 0], portion_index_matrix[:, 1], portion_index_matrix[:n_slow_count])
        #np.meshgrid fonksiyonu kullanarak olası tüm kombinasyonları hesaplıyoruz
        #Transpoze ederek satırlar ve sutunların yerlerini değiştirdik
        #reshape(-1, n_slow_continuous_index) kullanarak yavaş değişen değişken sayısı kadar sutun ve kendisinin belirleyeceği kadar satır oluşturulacak)
        #bu sayede bulduğumuz tüm olasılıklar return edilecek
        all_portion_index_combos = np.array(np.meshgrid(*portion_index_matrix),dtype=int).T.reshape(-1, n_slow_continuous_index)
        print("All portion index combos: {}".format(all_portion_index_combos))
        return all_portion_index_combos #örnek output = [[0 0], [0 1], [1 0], [1 1]] (2 değişken varsa)


    def make_slow_continuous_obs_index_floats(self, slow_continuous_index, cont_obs_index_floats):
        """
        :param slow_continuous_index: yavaş değişen değişenlerin indeksleri (self.slow_continuous_index
        :param cont_obs_index_floats:
        :return: slow_continuous_obs_index_floats yavaş ve sürekli değişen değişkenlerin gözlem endekslerini float şeklinde döndürüyoruz
        """
        slow_continuous_obs_index_floats = cont_obs_index_floats[: len(slow_continuous_index)]
        return slow_continuous_obs_index_floats


    def make_slow_continuous_obs_index_int_below(self, slow_continuous_obs_index_floats):
        """
        :param slow_continuous_obs_index_floats: yavaş ve sürekli değişen değişkenlerin gözlem endekslerini float şeklinde
        :return: slow_continuous_obs_index_int_below: yavaş değişen sürekli değişkenlerin ALT tam sayı değerleri
        """
        slow_continuous_obs_index_int_below = np.floor(slow_continuous_obs_index_floats).astype(np.int32) #integer şeklinde alt sayıları aldık
        return slow_continuous_obs_index_int_below


    def make_slow_continuous_obs_index_int_above(self, slow_continuous_obs_index_floats):
        """
        :param slow_continuous_obs_index_floats: yavaş ve sürekli değişen değişkenlerin gözlem endekslerini float şeklinde
        :return: slow_continuous_obs_index_int_above: yavaş değişen sürekli değişkenlerin ÜST tam sayı değerleri
        """
        slow_continuous_obs_index_int_above = np.ceil(slow_continuous_obs_index_floats).astype(np.int32) #integer şeklinde üst sayıları aldık
        return slow_continuous_obs_index_int_above


    def make_all_value_table_index_combos(self, slow_continuous_obs_index_int_below, slow_continuous_obs_index_int_above):
        #make_all_portion_index_combos bu fonksiyonumuz gibi :)
        """
        :param slow_continuous_obs_index_int_below: yavaş değişen sürekli değişkenlerin ALT tam sayı değerleri
        :param slow_continuous_obs_index_int_above: yavaş değişen sürekli değişkenlerin ÜST tam sayı değerleri
        :return: all_value_table_index_combos: alt ve üst sınır değerlerinin
        üstüste dizilip transpozunun alınmış hali açılıp tüm kombinasyonları reshape(-1, len(slow_cont_obs_index_int_above) ile döndürür.

        """
        value_table_index_matrix = np.vstack((slow_continuous_obs_index_int_below, slow_continuous_obs_index_int_above)).T
        # value_table_index_matrix = np.vstack ile alt ve üst sınırları alt alta dizilir daha sonrasında .T (transpoz) alımı ile satırlar ve sütunlar yer değiştirir

        all_value_table_index_combos = np.array(np.meshgrid(*value_table_index_matrix)).T.reshape(-1, len(slow_continuous_obs_index_int_below))
        #*portion_index_matrix  (yıldızı kullanarak) unpacking işlemi yapıyoruz. bu sayede her bir sütun ayrı ayrı np.meshgrid fonksiyonundan geçecek.
        #aslında şu anlama geliyor np.meshgrid(portion_index_matrix[:, 0], portion_index_matrix[:, 1], portion_index_matrix[:n_slow_count])
        #np.meshgrid fonksiyonu kullanarak olası tüm kombinasyonları hesaplıyoruz
        #Transpoze ederek satırlar ve sutunların yerlerini değiştirdik
        #reshape(-1, slow_continuous_obs_index_int_below) kullanarak yavaş değişen değişkenlerin alt(üst) sınır sayısı kadar sutun ve kendisinin belirleyeceği kadar satır oluşturulacak)
        #bu sayede bulduğumuz tüm olasılıklar return edilecek

        return all_value_table_index_combos


    def portion_calculator(self, slow_continuous_obs_index_int_above, slow_continuous_obs_index_floats):
        portion_below = slow_continuous_obs_index_int_above - slow_continuous_obs_index_floats #gözlemlenen değer ile alt sınır arasındaki fark
        portion_above = 1 - portion_below #gözlemnenen değer ile üst sınır arasındaki fark
        #alt ve üst sınırlar arasındaki farklar üst üste diziliyor ve daha sonrasında transpoze ediliyor (satır ve sütunlar yer değiştiriyor)
        portion_matrix = np.vstack((portion_below, portion_above)).T #ağırlık değerlerini içeren bir matris

        return portion_below, portion_above, portion_matrix