

# master_id: Eşsiz müşteri numarası
# order_channel: Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel: En son alışverişin yapıldığı kanal
# first_order_date: Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date: Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online: Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline: Müşterinin offline platformda yaptığı son alışveriş tarihi
##order_num_total_ever_online: Müşterinin online platformda yaptığı toplam alışveriş sayısı
## order_num_total_ever_offline: Müşterinin offline'da yaptığı toplam alışveriş sayısı
## customer_value_total_ever_offline: Müşterinin offline alışverişlerinde ödediği toplam ücret
## customer_value_total_ever_online: Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12: Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi



import pandas as pd
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None) '''bu islem tum satirlari da gormek istediginde yapilir.'''
pd.set_option('display.float_format' , lambda x: '%.4f' % x)
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler


# ###############################################################
# Görev 1: Veriyi Hazırlama
# ###############################################################

# Adım 1: flo_data_20K.csv verisini okuyunuz.

df_ = pd.read_csv('/Users/rmarabaci/PycharmProjects/pythonProject1/data_science_bootcamp/Hafta_4_CRM_Analytics/FLO_RFM_Analizi/flo_data_20k.csv')

df = df_.copy()

df.head()


# Adım 2: Aykırı değerleri baskılamak için gerekli olan outlier_thresholds ve replace_with_thresholds fonksiyonlarını tanımlayınız.
# Not: cltv hesaplanırken frequency değerleri integer olması gerekmektedir.Bu nedenle alt ve üst limitlerini round() ile yuvarlayınız.

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    up_limit = (up_limit)
    low_limit = (low_limit)
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = round(low_limit, 0)
    dataframe.loc[(dataframe[variable] > up_limit), variable] = round(up_limit, 0)


# Adım 3: "order_num_total_ever_online", "order_num_total_ever_offline", "customer_value_total_ever_offline",
# "customer_value_total_ever_online" değişkenlerinin aykırı değerleri varsa baskılayanız.

baskilama_listesi = ["order_num_total_ever_online",
          "order_num_total_ever_offline",
          "customer_value_total_ever_offline",
          "customer_value_total_ever_online"]

df.describe().T

for col in baskilama_listesi:
    replace_with_thresholds(df, col)

df.describe().T


# Adım 4: Omnichannel müşterilerin hem online'dan hem de offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.

df["order_num_total"] = df["order_num_total_ever_online"] + df["order_num_total_ever_offline"]
df["customer_value_total"] = df["customer_value_total_ever_offline"] + df["customer_value_total_ever_online"]

df.head()

# Adım 5: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz

df.dtypes

df['first_order_date'] = pd.to_datetime(df['first_order_date'])

df['last_order_date'] = pd.to_datetime(df['last_order_date'])

df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

df.dtypes

# ###############################################################
# Görev 2: CLTV Veri Yapısının Oluşturulması
# ###############################################################

# Adım 1: Veri setindeki en son alışverişin yapıldığı tarihten 2 gün sonrasını analiz tarihi olarak alınız.

df['last_order_date'].max()
today_date = dt.datetime(2021, 6, 1)
type(today_date)

df.head()

# Adım 2: customer_id, recency_cltv_weekly, T_weekly, frequency ve monetary_cltv_avg değerlerinin yer aldığı yeni bir cltv dataframe'i oluşturunuz.
# Monetary değeri satın alma başına ortalama değer olarak, recency ve tenure değerleri ise haftalık cinsten ifade edilecek.

# recency: Son satın alma üzerinden geçen zaman. Haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı. Haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# frequency: tekrar eden toplam satın alma sayısı (frequency>1)
# monetary: satın alma başına ortalama kazanç


cltv_df = pd.DataFrame()

cltv_df["customer_id"] = df["master_id"]

cltv_df["recency_cltv_weekly"] = ((df["last_order_date"]- df["first_order_date"]).astype('timedelta64[D]')) / 7

cltv_df["T_weekly"] = ((today_date - df["first_order_date"]).astype('timedelta64[D]'))/7

cltv_df["frequency"] = df["order_num_total"]

cltv_df["monetary_cltv_avg"] = df["customer_value_total"] / df["order_num_total"]



# ###############################################################
# Görev 3: BG/NBD, Gamma-Gamma Modellerinin Kurulması ve CLTV’nin Hesaplanması
# ###############################################################

# Adım 1: BG/NBD modelini fit ediniz.

bgf = BetaGeoFitter(penalizer_coef= 0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df["recency_cltv_weekly"],
        cltv_df["T_weekly"])

# • 3 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_3_month olarak cltv
# dataframe'ine ekleyiniz.

cltv_df["exp_sales_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*3,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])
#  6 ay içerisinde müşterilerden beklenen satın almaları tahmin ediniz ve exp_sales_6_month olarak cltv
# dataframe'ine ekleyiniz

cltv_df["exp_sales_6_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4*6,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency_cltv_weekly"],
                                                        cltv_df["T_weekly"])


# Adım 2: Gamma-Gamma modelini fit ediniz. Müşterilerin ortalama bırakacakları değeri tahminleyip exp_average_value olarak cltv
# dataframe'ine ekleyiniz.

ggf = GammaGammaFitter(penalizer_coef=0.01)

ggf.fit(cltv_df['frequency'], cltv_df["monetary_cltv_avg"])

cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                       cltv_df["monetary_cltv_avg"])


cltv_df.sort_values("exp_average_value", ascending=False)

# Adım 3: 6 aylık CLTV hesaplayınız ve cltv ismiyle dataframe'e ekleyiniz.
# • Cltv değeri en yüksek 20 kişiyi gözlemleyiniz

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency_cltv_weekly'],
                                   cltv_df['T_weekly'],
                                   cltv_df['monetary_cltv_avg'],
                                   time=6,  # 6 aylık (burada time ay cinsindendir)
                                   freq="W",  # T'nin frekans bilgisi. Biz yukarida haftalik calistik o sebeple 'W'
                                   discount_rate=0.01) #ilerleyen donemlerde fiyatlarda indirim yapma durumu icin

cltv_df["cltv"] = cltv

cltv_df.sort_values("cltv",ascending=False)[:20]


# ###############################################################
# Görev 4: CLTV Değerine Göre Segmentlerin Oluşturulması
# ###############################################################

# Adım 1: 6 aylık CLTV'ye göre tüm müşterilerinizi 4 gruba (segmente) ayırınız ve grup isimlerini veri setine ekleyiniz.

cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_df.head()

# Adım 2: 4 grup içerisinden seçeceğiniz 2 grup için yönetime kısa kısa 6 aylık aksiyon önerilerinde bulununuz.


